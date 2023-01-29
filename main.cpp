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

    template<typename Scalar>
    struct Node {
        Eigen::Vector<Scalar, pose_state_dim> x, xd;

        Node() :
            x{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, M_PI_4, -M_PI_2, 0.0, M_PI_4, -M_PI_2},
            xd{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} {
        }

        template<typename Vector>
        Node(Vector&& x, Vector&& xd) : x{x}, xd{xd} {
        }
    };

    template<typename Scalar>
    struct State {
        Eigen::Vector<Scalar, pose_state_dim> p;
        Eigen::Vector<Scalar, state_dim> v, a;
    };

    template<typename Scalar>
    class Polynomial {
    public:
        using ADScalar = CppAD::AD<Scalar>;

        explicit Polynomial(const Scalar& dt) : dt{dt} {
            make_fun();
        }

        void update_nodes(const Node<Scalar>& head, const Node<Scalar>& tail) {
            Eigen::VectorX<Scalar> n(4 * pose_state_dim);
            n << head.x, head.xd, tail.x, tail.xd;

            fun_p.new_dynamic(n);
            fun_v.new_dynamic(n);
        }

        State<Scalar> eval(const Scalar& t) {
            State<Scalar> state;

            Eigen::VectorX<Scalar> vec_t(1);
            vec_t << t;

            state.p << fun_p.Forward(0, vec_t);
            state.v << fun_v.Forward(0, vec_t);
            state.a << fun_v.Jacobian(vec_t);

            return state;
        }

    private:
        const Scalar dt;

        CppAD::ADFun<Scalar> fun_p, fun_v;

        void make_fun() {
            Eigen::VectorX<ADScalar> t(1);
            Eigen::VectorX<ADScalar> n(4 * pose_state_dim);

            Eigen::VectorX<ADScalar> p(pose_state_dim);
            Eigen::VectorX<ADScalar> v(state_dim);

            CppAD::Independent(t, n);

            Node<ADScalar> head{
                n.segment(0, pose_state_dim),
                n.segment(pose_state_dim, pose_state_dim)
            };
            Node<ADScalar> tail{
                n.segment(2 * pose_state_dim, pose_state_dim),
                n.segment(3 * pose_state_dim, pose_state_dim)
            };

            Eigen::VectorX<ADScalar> a2 = -Scalar(1.0) / (dt * dt) *
                (Scalar(3.0) * (head.x - tail.x) + dt * (Scalar(2.0) * head.xd + tail.xd));
            Eigen::VectorX<ADScalar> a3 = Scalar(1.0) / (dt * dt * dt) *
                (Scalar(2.0) * (head.x - tail.x) + dt * (head.xd + tail.xd));

            p = head.x.template cast<ADScalar>()
                + head.xd.template cast<ADScalar>() * t(0)
                + a2.template cast<ADScalar>() * (t(0) * t(0))
                + a3.template cast<ADScalar>() * (t(0) * t(0) * t(0));

            fun_p.Dependent(t, p);
            auto fun_ad_p = fun_p.base2ad();

            CppAD::Independent(t, n);

            fun_ad_p.new_dynamic(n);
            p = fun_ad_p.Forward(0, t);
            Eigen::VectorX<ADScalar> pd = fun_ad_p.Jacobian(t);

            Eigen::Quaternion<ADScalar> r(p.template head<4>());
            Eigen::Quaternion<ADScalar> rd(pd.template head<4>());

            Eigen::Quaternion<ADScalar> w = rd * r.conjugate();
            v << Scalar(2.0) * w.coeffs().tail(3) / r.squaredNorm(), pd.tail(state_dim - 3);

            fun_v.Dependent(t, v);
        }
    };
}

int main() {
    using namespace biped::rcg;

    to::Polynomial<Scalar> polynomial(10.0);
    to::Node<Scalar> head, tail;

    tail.x.head<4>() << 0.0, 1.0, 0.0, 0.0;
    tail.xd << -1.0, 1.0, 0.5, -0.75, 2.0, -3.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0;

    polynomial.update_nodes(head, tail);
    auto state = polynomial.eval(10.0);

    std::cout << state.p.transpose() << '\n'
              << state.v.transpose() << '\n'
              << state.a.transpose() << '\n';

    return 0;
}
