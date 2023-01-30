#include <iostream>
#include <vector>

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

    enum Derivative {
        POSITION = 0,
        VELOCITY,
        ACCELERATION
    };

    enum Component {
        BASE_ANGULAR = 0,
        BASE_LINEAR,
        JOINT_STATE
    };

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

    template<typename Scalar, int N>
    struct Node {
        Eigen::Vector<Scalar, N> x, xd;
    };

    template<typename Scalar, int N>
    class Polynomial {
    public:
        using ADScalar = CppAD::AD<Scalar>;

        explicit Polynomial(const Scalar& dt) : dt{dt} {
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
            x << ADScalar(1.0), t(0), t(0) * t(0), t(0) * t(0) * t(0);

            p = a * x;
            fun_p.Dependent(p);
            fun_p.optimize("no_compare_op");

            auto fun_ad_p = fun_p.base2ad();
            CppAD::Independent(t, n);

            fun_ad_p.new_dynamic(n);
            pd << fun_ad_p.Jacobian(t);

            fun_pd.Dependent(pd);
            fun_pd.optimize("no_compare_op");
        }

        virtual void update_nodes(const Node<Scalar, N>& head, const Node<Scalar, N>& tail) {
            Eigen::VectorX<Scalar> n(4 * N);
            n << head.x, head.xd, tail.x, tail.xd;
            fun_p.new_dynamic(n);
            fun_pd.new_dynamic(n);
        }

        virtual Eigen::VectorX<Scalar> eval(const Scalar& t, int n) {
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

    protected:
        const Scalar dt;
        CppAD::ADFun<Scalar> fun_p, fun_pd;
    };

    template<typename Scalar>
    class QuaternionPolynomial : public Polynomial<Scalar, 4> {
    public:
        using Base = Polynomial<Scalar, 4>;
        using ADScalar = CppAD::AD<Scalar>;

        static constexpr int N = 4, M = 3;

        explicit QuaternionPolynomial(const Scalar& dt) : Base(dt) {
            Eigen::VectorX<ADScalar> t(1), n(4 * N), v(M);

            ADScalar _0{0.0}, _1{1.0};
            n << _1, _0, _0, _0,
                _0, _0, _0, _0,
                _1, _0, _0, _0,
                _0, _0, _0, _0;

            Eigen::Vector4<ADScalar> p, pd;

            CppAD::Independent(t, n);

            auto fun_ad_p = Base::fun_p.base2ad();
            fun_ad_p.new_dynamic(n);

            p << fun_ad_p.Forward(0, t);
            pd << fun_ad_p.Jacobian(t);

            Eigen::Quaternion<ADScalar> r{p}, rd{pd}, r_rd;
            r_rd = rd * r.conjugate();

            v << Scalar(2.0) * r_rd.coeffs().tail(M) / r.squaredNorm();
            fun_v.Dependent(v);
            fun_v.optimize("no_compare_op");
        }

        void update_nodes(const Node<Scalar, N>& head, const Node<Scalar, N>& tail) override {
            Eigen::VectorX<Scalar> n(4 * N);
            n << head.x, head.xd, tail.x, tail.xd;
            Base::fun_p.new_dynamic(n);
            fun_v.new_dynamic(n);
        }

        Eigen::VectorX<Scalar> eval(const Scalar& t, int n) override {
            Eigen::VectorX<Scalar> vt(1);
            vt << t;

            switch (n) {
                case 0:
                    return Base::fun_p.Forward(0, vt);
                case 1:
                    return fun_v.Forward(0, vt);
                case 2:
                    return fun_v.Jacobian(vt);
                default:
                    assert(false);
            }
        }

    private:
        CppAD::ADFun<Scalar> fun_v;
    };
}

int main() {
    using namespace biped::rcg;

    to::Polynomial<Scalar, 4> polynomial(10.0);

    to::Node<Scalar, 4> head{Eigen::Vector4<Scalar>{1.0, 0.0, 0.0, 0.0}, Eigen::Vector4<Scalar>{0.0, 0.0, 0.0, 0.0}};
    to::Node<Scalar, 4> tail{Eigen::Vector4<Scalar>{0.0, 1.0, 0.0, 0.0}, Eigen::Vector4<Scalar>{0.0, 0.0, 0.0, 0.0}};

    polynomial.update_nodes(head, tail);

    Scalar t{0.25};

    std::cout << polynomial.eval(t, 0).transpose() << '\n';
    std::cout << polynomial.eval(t, 1).transpose() << '\n';
    std::cout << polynomial.eval(t, 2).transpose() << '\n';

    to::QuaternionPolynomial<Scalar> quaternion_polynomial(10.0);
    quaternion_polynomial.update_nodes(head, tail);

    std::cout << quaternion_polynomial.eval(t, 0).transpose() << '\n';
    std::cout << quaternion_polynomial.eval(t, 1).transpose() << '\n';
    std::cout << quaternion_polynomial.eval(t, 2).transpose() << '\n';

    return 0;
}
