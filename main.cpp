#include <iostream>
#include <vector>
#include <limits>
#include <cmath>

#include <Eigen/Geometry>

#include "biped/declarations.h"
#include "biped/transforms.h"
#include "biped/jacobians.h"
#include "biped/inertia_properties.h"
#include "biped/inverse_dynamics.h"
#include "biped/jsim.h"

namespace to {
    using namespace biped::rcg;

    static constexpr int pose_state_dim = 7 + JointSpaceDimension;
    static constexpr int state_dim = 6 + JointSpaceDimension;

    template<typename Scalar, int N>
    class Polynomial;

    template<typename Scalar>
    class RotationAdaptor;

    struct Model {
        Model() :
            inverse_dynamics(inertia_properties, motion_transforms),
            jsim(inertia_properties, force_transforms) {
        }

        HomogeneousTransforms homogeneous_transforms;
        MotionTransforms motion_transforms;
        ForceTransforms force_transforms;
        Jacobians jacobians;
        InertiaProperties inertia_properties;

        InverseDynamics inverse_dynamics;
        JSIM jsim;
    };

    template<typename Scalar, int N, int T = N>
    struct Node {
        Eigen::Vector<Scalar, N> x;
        Eigen::Vector<Scalar, T> xd;
    };

    template<typename Scalar, int N>
    class Polynomial {
    public:
        friend class RotationAdaptor<Scalar>;

        using ADScalar = CppAD::AD<Scalar>;

        explicit Polynomial(const Scalar& dt) {
            Eigen::VectorX<ADScalar> t(1), n(4 * N), p(N), pd(N);

            CppAD::Independent(t, n);

            Node<ADScalar, N> head{n.segment(0, N), n.segment(N, N)};
            Node<ADScalar, N> tail{n.segment(2 * N, N), n.segment(3 * N, N)};

            Eigen::Matrix<ADScalar, N, 4> a;
            ADScalar _1{1.0}, _2{2.0}, _3{3.0};

            a.col(0) << head.x;
            a.col(1) << head.xd;
            a.col(2) << -_1 / (dt * dt) * (_3 * (head.x - tail.x) + dt * (_2 * head.xd + tail.xd));
            a.col(3) << _1 / (dt * dt * dt) * (_2 * (head.x - tail.x) + dt * (head.xd + tail.xd));

            Eigen::Vector4<ADScalar> x;
            x << _1, t(0), t(0) * t(0), t(0) * t(0) * t(0);

            p = a * x;
            fun_p.Dependent(p);
//            fun_p.optimize("no_compare_op");

            CppAD::Independent(t, n);
            auto fun_ad_p = fun_p.base2ad();
            fun_ad_p.new_dynamic(n);

            pd << fun_ad_p.Jacobian(t);
            fun_pd.Dependent(pd);
//            fun_pd.optimize("no_compare_op");
        }

        void update_nodes(const Node<Scalar, N>& head, const Node<Scalar, N>& tail) {
            Eigen::VectorX<Scalar> n(4 * N);
            n << head.x, head.xd, tail.x, tail.xd;
            fun_p.new_dynamic(n);
            fun_pd.new_dynamic(n);
        }

        Eigen::VectorX<Scalar> eval(const Scalar& t, int n) {
            Eigen::VectorX<Scalar> vt(1);
            vt << t;

            switch (n) {
                case 0:
                    return fun_p.Forward(0, vt);
                case 1:
                    return fun_pd.Forward(0, vt);
                case 2:
                    return fun_pd.Jacobian(vt);
                default:
                    assert(false);
            }
        }

    private:
        CppAD::ADFun<Scalar> fun_p, fun_pd;
    };

    template<typename Scalar>
    class RotationAdaptor {
    public:
        using ADScalar = CppAD::AD<Scalar>;
        static constexpr int N = 4, T = 3;

        Polynomial<Scalar, T> polynomial;

        explicit RotationAdaptor(const Scalar& dt) : polynomial(dt) {
            Eigen::VectorX<ADScalar> t(1), n(4 * T), q_c(4), w(3);
            Eigen::Vector3<ADScalar> v, dv;

            CppAD::Independent(t, n);
            auto fun_ad_p = polynomial.fun_p.base2ad();
            fun_ad_p.new_dynamic(n);
            v << fun_ad_p.Forward(0, t);

            q_c << exp(v);
            fun_q.Dependent(q_c);

            CppAD::Independent(t, n);
            auto fun_ad_q = fun_q.base2ad();
            fun_ad_q.new_dynamic(n);

            Eigen::Vector4<ADScalar> qd_c4, q_c4;
            q_c4 << fun_ad_q.Forward(0, t);
            qd_c4 << fun_ad_q.Jacobian(t);
            Eigen::Quaternion<ADScalar> qd{qd_c4}, q{q_c4};

            ADScalar _2{2.0};
            w << _2 * (qd * q.inverse()).coeffs().tail(3);
            fun_w.Dependent(w);
        }

        void update_nodes(const Node<Scalar, T>& head, const Node<Scalar, T>& tail) {
            polynomial.update_nodes(head, tail);

            Eigen::VectorX<Scalar> n(4 * T);
            n << head.x, head.xd, tail.x, tail.xd;
            fun_q.new_dynamic(n);
            fun_w.new_dynamic(n);
        }

        Eigen::VectorX<Scalar> eval(const Scalar& t, int n) {
            Eigen::VectorX<Scalar> vt(1);
            vt << t;

            switch (n) {
                case 0:
                    return fun_q.Forward(0, vt);
                case 1:
                    return fun_w.Forward(0, vt);
                case 2:
                    return fun_w.Jacobian(vt);
                default:
                    assert(false);
            }
        }

    private:
        Eigen::Quaternion<Scalar> r0;
        CppAD::ADFun<Scalar> fun_q, fun_w;

        static Eigen::Vector4<ADScalar> exp(const Eigen::Vector3<ADScalar> v) {
            constexpr double eps = 0.00012207;
            ADScalar _1_2{0.5}, _1{1.0}, _8{8.0}, _48{48.0}, _384{384.0}, _eps{eps};

            ADScalar t2 = v.squaredNorm();
            ADScalar t4 = t2 * t2;
            ADScalar t = CppAD::sqrt(t2);
            ADScalar st = CppAD::sin(_1_2 * t);
            ADScalar ct = CppAD::CondExpLe(t, _eps, _1 - t2 / _8 + t4 / _384, CppAD::cos(_1_2 * t));
            ADScalar sc = CppAD::CondExpLe(t, _eps, _1_2 + t2 / _48, st / t);

            Eigen::Vector3<ADScalar> sv = sc * v;
            return {sv(0), sv(1), sv(2), ct};
        }
    };
}

int main() {
    using namespace biped::rcg;

    Scalar t{5.0};

    {
        to::Polynomial<Scalar, 4> polynomial(10.0);

        to::Node<Scalar, 4> head{Eigen::Vector4<Scalar>{1.0, 0.0, 0.0, 0.0}, Eigen::Vector4<Scalar>{0.0, 0.0, 0.0, 0.0}};
        to::Node<Scalar, 4> tail{Eigen::Vector4<Scalar>{0.0, 1.0, 0.0, 0.0}, Eigen::Vector4<Scalar>{0.0, 0.0, 0.0, 0.0}};

        polynomial.update_nodes(head, tail);

        std::cout << polynomial.eval(t, 0).transpose() << '\n';
        std::cout << polynomial.eval(t, 1).transpose() << '\n';
        std::cout << polynomial.eval(t, 2).transpose() << '\n';
    }

    {
        to::Node<Scalar, 3> head{Eigen::Vector3<Scalar>{1.0, 0.0, 0.0}, Eigen::Vector3<Scalar>{0.0, 0.0, 0.0}};
        to::Node<Scalar, 3> tail{Eigen::Vector3<Scalar>{0.0, 1.0, 0.0}, Eigen::Vector3<Scalar>{0.0, 0.0, 0.0}};

        to::RotationAdaptor<Scalar> rotation_adaptor(10.0);
        rotation_adaptor.update_nodes(head, tail);

        std::cout << rotation_adaptor.eval(t, 0).transpose() << '\n';
        std::cout << rotation_adaptor.eval(t, 1).transpose() << '\n';
        std::cout << rotation_adaptor.eval(t, 2).transpose() << '\n';
    }
    return 0;
}
