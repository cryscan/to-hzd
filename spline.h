//
// Created by cryscan on 1/30/23.
//

#ifndef TO_HZD_SPLINE_H
#define TO_HZD_SPLINE_H

#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <cppad/cppad.hpp>

#include "forward.h"

namespace to {
    template<typename Scalar>
    struct Node {
        Eigen::VectorX<Scalar> x;
        Eigen::VectorX<Scalar> xd;
    };

    template<typename Scalar>
    class Polynomial {
        friend class RotationAdaptor<Scalar>;

    public:
        using ADScalar = CppAD::AD<Scalar>;
        const int D;

        Polynomial(const Scalar& dt, int d)
            : D{d} {
            Eigen::VectorX<ADScalar> t(1), x(4 * d), p(d), pd(d);

            CppAD::Independent(t, x);

            Node<ADScalar> head{x.segment(0, d), x.segment(d, d)};
            Node<ADScalar> tail{x.segment(2 * d, d), x.segment(3 * d, d)};

            Eigen::MatrixX4<ADScalar> a(d, 4);
            ADScalar _1{1.0}, _2{2.0}, _3{3.0};

            a.col(0) << head.x;
            a.col(1) << head.xd;
            a.col(2) << -_1 / (dt * dt) * (_3 * (head.x - tail.x) + dt * (_2 * head.xd + tail.xd));
            a.col(3) << _1 / (dt * dt * dt) * (_2 * (head.x - tail.x) + dt * (head.xd + tail.xd));

            Eigen::Vector4<ADScalar> vt{_1, t(0), t(0) * t(0), t(0) * t(0) * t(0)};
            p = a * vt;
            fun_p.Dependent(p);
            if (d > 3) fun_p.optimize("no_compare_op");

            CppAD::Independent(t, x);
            auto fun_ad_p = fun_p.base2ad();
            fun_ad_p.new_dynamic(x);

            pd << fun_ad_p.Jacobian(t);
            fun_pd.Dependent(pd);
            if (d > 3) fun_pd.optimize("no_compare_op");
        }

        void update_nodes(const Node<Scalar>& head, const Node<Scalar>& tail) {
            Eigen::VectorX<Scalar> x(4 * D);
            x << head.x, head.xd, tail.x, tail.xd;
            fun_p.new_dynamic(x);
            fun_pd.new_dynamic(x);
        }

        Eigen::VectorX<Scalar> eval(const Scalar& t, int n) {
            Eigen::VectorX<Scalar> vt = Eigen::Vector<Scalar, 1>{t};
            switch (n) {
                case 0: return fun_p.Forward(0, vt);
                case 1: return fun_pd.Forward(0, vt);
                case 2: return fun_pd.Jacobian(vt);
                default: assert(false);
            }
        }

    private:
        CppAD::ADFun<Scalar> fun_p, fun_pd;
    };

    template<typename Scalar>
    class RotationAdaptor {
    public:
        using ADScalar = CppAD::AD<Scalar>;
        static constexpr int D = 3;

        explicit RotationAdaptor(const Scalar& dt, int d = D)
            : polynomial(dt, D) {
            assert(d == D);
            Eigen::VectorX<ADScalar> t(1), x(4 * D), q(4), w(3);

            CppAD::Independent(t, x);
            auto fun_ad_p = polynomial.fun_p.base2ad();
            fun_ad_p.new_dynamic(x);

            Eigen::Vector3<ADScalar> v = fun_ad_p.Forward(0, t);
            q << exp(v);

            fun_q.Dependent(q);
            fun_q.optimize("no_compare_op");

            CppAD::Independent(t, x);
            auto fun_ad_q = fun_q.base2ad();
            fun_ad_q.new_dynamic(x);

            Eigen::Vector4<ADScalar> q4 = fun_ad_q.Forward(0, t);
            Eigen::Vector4<ADScalar> qd = fun_ad_q.Jacobian(t);
            Eigen::Quaternion<ADScalar> r{q4}, rd{qd};

            ADScalar _2{2.0};
            w << _2 * (rd * r.inverse()).coeffs().tail(3);
            fun_w.Dependent(w);
            fun_w.optimize("no_compare_op");
        }

        void update_nodes(const Node<Scalar>& head, const Node<Scalar>& tail) {
            polynomial.update_nodes(head, tail);

            Eigen::VectorX<Scalar> x(4 * D);
            x << head.x, head.xd, tail.x, tail.xd;
            fun_q.new_dynamic(x);
            fun_w.new_dynamic(x);
        }

        Eigen::VectorX<Scalar> eval(const Scalar& t, int n) {
            Eigen::VectorX<Scalar> vt = Eigen::Vector<Scalar, 1>{t};
            switch (n) {
                case 0: return fun_q.Forward(0, vt);
                case 1: return fun_w.Forward(0, vt);
                case 2: return fun_w.Jacobian(vt);
                default: assert(false);
            }
        }

        Eigen::VectorX<Scalar> eval_qd(const Scalar& t) {
            Eigen::VectorX<Scalar> vt = Eigen::Vector<Scalar, 1>{t};
            return fun_q.Jacobian(vt);
        }

    private:
        Polynomial<Scalar> polynomial;
        CppAD::ADFun<Scalar> fun_q, fun_w;

        // Exponential map that maps R^3 to S^3.
        // Reference: https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf
        static Eigen::Vector4<ADScalar> exp(const Eigen::Vector3<ADScalar> v) {
            ADScalar _1_2{0.5}, _1{1.0}, _8{8.0}, _48{48.0}, _384{384.0};
            ADScalar eps{0.00012207};

            ADScalar t2 = v.squaredNorm();
            ADScalar t4 = t2 * t2;
            ADScalar t = CppAD::sqrt(t2);
            ADScalar st = CppAD::sin(_1_2 * t);
            ADScalar ct = CppAD::CondExpLe(t, eps, _1 - t2 / _8 + t4 / _384, CppAD::cos(_1_2 * t));
            ADScalar sc = CppAD::CondExpLe(t, eps, _1_2 + t2 / _48, st / t);

            Eigen::Vector3<ADScalar> sv = sc * v;
            return {sv(0), sv(1), sv(2), ct};
        }
    };

    template<typename Scalar, PolynomialType<Scalar> Polynomial>
    class Spline {
    public:
        using Nodes = std::vector<Node<Scalar>>;
        using Polynomials = std::vector<Polynomial>;
        const int D;

        Spline(Nodes nodes, const Scalar& dt, int d)
            : nodes{std::move(nodes)},
              dt{dt},
              D{d} {
            make_polynomials();
        }

        [[nodiscard]] Scalar total_time() const {
            return Scalar{polynomials.size()} * dt;
        }

        [[nodiscard]] auto polynomial_id(const Scalar& t) const {
            return CppAD::Integer(t / dt);
        }

        [[nodiscard]] Scalar local_time(const Scalar& t) const {
            return t - polynomial_id(t) * dt;
        }

        void update_nodes(const Nodes& nodes_) {
            assert(nodes.size() == nodes_.size());
            nodes = nodes_;
            update_polynomials();
        }

        void update_node_at(auto i, const Node<Scalar>& node) {
            nodes[i] = node;
            update_polynomials_at_node(i);
        }

        auto eval(const Scalar& t, auto n) {
            auto i = polynomial_id(t);
            return polynomials[i].eval(local_time(t), n);
        }

    private:
        Nodes nodes;
        Polynomials polynomials;
        const Scalar dt;

        void make_polynomials() {
            assert(nodes.size() >= 2);

            polynomials.clear();
            polynomials.reserve(nodes.size() - 1);
            for (int i = 0; i < nodes.size() - 1; ++i) {
                polynomials.emplace_back(dt, D);
                polynomials.back().update_nodes(nodes[i], nodes[i + 1]);
            }
        }

        void update_polynomials() {
            for (int i = 0; i < nodes.size() - 1; ++i) {
                polynomials[i].update_nodes(nodes[i], nodes[i + 1]);
            }
        }

        void update_polynomials_at_node(auto i) {
            if (i < nodes.size())
                polynomials[i].update_nodes(nodes[i], nodes[i + 1]);
            if (i > 0)
                polynomials[i - 1].update_nodes(nodes[i - 1], nodes[i]);
        }
    };
}// namespace to

#endif//TO_HZD_SPLINE_H
