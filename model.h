//
// Created by cryscan on 1/30/23.
//

#ifndef TO_HZD_MODEL_H
#define TO_HZD_MODEL_H

#include "biped/declarations.h"
#include "biped/inertia_properties.h"
#include "biped/inverse_dynamics.h"
#include "biped/jacobians.h"
#include "biped/jsim.h"
#include "biped/transforms.h"

#include "forward.h"
#include "spline.h"
#include "utils.h"

namespace to {
    static constexpr int joint_space_dim = Biped::rcg::JointSpaceDimension;
    static constexpr int pose_state_dim = 7 + joint_space_dim;
    static constexpr int state_dim = 6 + joint_space_dim;

    enum Side {
        LEFT = 0,
        RIGHT,
        NONE
    };

    struct State {
        using Scalar = Biped::rcg::Scalar;

        Biped::rcg::JointState q, qd, qdd;
        Eigen::Quaternion<Scalar> r;
        Biped::rcg::Vector3 v, vd;     // Base angular velocity/acceleration (body frame)
        Biped::rcg::Vector3 p, pd, pdd;// Base linear pose/velocity/acceleration (inertial frame)

        Biped::rcg::JointState u;// Control
        Biped::rcg::Vector3 f;   // Contact force
        Side side = NONE;

        [[nodiscard]] auto state_vector() const {
            Eigen::Vector<Scalar, pose_state_dim> x;
            x << r.coeffs(), p, q;
            return x;
        }

        [[nodiscard]] auto base_body_velocity() const {
            Biped::rcg::Velocity xd;
            auto rot = r.inverse().toRotationMatrix();
            xd << v, rot * pd;
            return xd;
        }

        [[nodiscard]] auto body_velocity() const {
            Eigen::Vector<Scalar, state_dim> xd;
            xd << base_body_velocity(), qd;
            return xd;
        }

        [[nodiscard]] auto state_derivative() const {
            Eigen::Vector<Scalar, pose_state_dim> xd;
            xd << body_to_quaternion_derivative(r, v).coeffs(), pd, qd;
            return xd;
        }

        [[nodiscard]] auto base_acceleration() const {
            Biped::rcg::Acceleration xdd;
            auto rot = r.inverse().toRotationMatrix();
            xdd << vd, rot * pdd;
            return xdd;
        }

        [[nodiscard]] auto body_acceleration() const {
            Eigen::Vector<Scalar, state_dim> xdd;
            xdd << base_acceleration(), qdd;
            return xdd;
        }
    };

    const Biped::rcg::Acceleration gravity{0.0, 0.0, 0.0, 0.0, 0.0, -Biped::rcg::g};

    class Model {
    public:
        using Scalar = Biped::rcg::Scalar;

        Model()
            : inverse_dynamics(inertia_properties, motion_transforms),
              jsim(inertia_properties, force_transforms) {
        }

        Eigen::Vector<Scalar, state_dim> dynamics_constraint(const State& state) {
            using namespace Biped::rcg;
            Eigen::Vector<Scalar, state_dim> u;

            u << Force::Zero(), state.u;
            auto h = nonlinear_terms(state);
            auto contact_jacobian = foot_jacobian(state).bottomRows<3>();

            return jsim(state.q) * state.body_acceleration() + h - contact_jacobian.transpose() * state.f - u;
        }

        Biped::rcg::Vector3 contact_constraint(const State& state) {
            using namespace Biped::rcg;

            auto contact_jacobian = foot_jacobian(state).bottomRows<3>();
            auto contact_jacobian_derivative = foot_jacobian_time_derivative(state).bottomRows<3>();

            return contact_jacobian * state.body_acceleration() + contact_jacobian_derivative * state.state_derivative();
        }

    private:
        Biped::rcg::HomogeneousTransforms homogeneous_transforms;
        Biped::rcg::MotionTransforms motion_transforms;
        Biped::rcg::ForceTransforms force_transforms;
        Biped::rcg::Jacobians jacobians;
        Biped::rcg::InertiaProperties inertia_properties;

        Biped::rcg::InverseDynamics inverse_dynamics;
        Biped::rcg::JSIM jsim;

        Eigen::Matrix<Scalar, 6, state_dim> foot_jacobian(const State& state) {
            Eigen::Matrix<Scalar, 6, joint_space_dim> joint_space_jacobian = decltype(joint_space_jacobian)::Zero();
            Eigen::Vector4<Scalar> p{0, 0, 0, 1};
            switch (state.side) {
                case Side::LEFT:
                    joint_space_jacobian.leftCols<3>() = jacobians.fr_trunk_J_L_foot(state.q);
                    p = homogeneous_transforms.fr_trunk_X_L_foot(state.q) * p;
                    break;
                case Side::RIGHT:
                    joint_space_jacobian.rightCols<3>() = jacobians.fr_trunk_J_R_foot(state.q);
                    p = homogeneous_transforms.fr_trunk_X_R_foot(state.q) * p;
                    break;
                case Side::NONE:
                    break;
            }

            auto px = hat<Scalar>(p.head<3>());

            // J =  [   R       ,   0   ,   R Jr    ]
            //      [   -R px   ,   R   ,   R Jp    ]
            Eigen::Matrix<Scalar, 6, state_dim> jacobian = decltype(jacobian)::Zero();
            Eigen::Matrix3<Scalar> r = state.r.toRotationMatrix();
            jacobian.block<3, 3>(0, 0) = r;      // Base angular -> EE angular
            jacobian.block<3, 3>(3, 0) = -r * px;// Base angular -> EE linear
            jacobian.block<3, 3>(3, 3) = r;      // Base linear  -> EE linear
            jacobian.block<3, joint_space_dim>(0, 6) = r * joint_space_jacobian.topRows<3>();
            jacobian.block<3, joint_space_dim>(3, 6) = r * joint_space_jacobian.bottomRows<3>();

            return jacobian;
        }

        template<typename Vector>
        Eigen::Matrix<Scalar, 6, state_dim> foot_jacobian(const Vector& x, Side side) {
            State state;

            state.r = Eigen::Quaternion<Scalar>(x(3), x(0), x(1), x(2));
            state.p = x.segment(4, 3);
            state.q = x.tail(joint_space_dim);
            state.side = side;

            return foot_jacobian(state);
        }

        Eigen::Vector<Scalar, state_dim> nonlinear_terms(const State& state) {
            using namespace Biped::rcg;
            Eigen::Vector<Scalar, state_dim> h;

            Force h_base;
            JointState h_joint;

            inverse_dynamics.id_fully_actuated(
                h_base,
                h_joint,
                gravity,
                state.base_body_velocity(),
                Acceleration::Zero(),
                state.q,
                state.qd,
                JointState::Zero());

            h << h_base, h_joint;
            return h;
        }

        Eigen::Matrix<Scalar, 6, pose_state_dim> foot_jacobian_time_derivative(const State& state) {
            Eigen::VectorX<Scalar> x(pose_state_dim), y(6);

            CppAD::Independent(x);

            auto contact_jacobian = foot_jacobian(x, state.side);
            y = contact_jacobian * state.body_velocity();

            CppAD::ADFun<double> fun;
            fun.Dependent(x, y);
            auto fun_ad = fun.base2ad();

            Eigen::Matrix<Scalar, 6, pose_state_dim> jacobian_derivative;
            x = state.state_vector();
            jacobian_derivative << fun_ad.Jacobian(x);
            return jacobian_derivative;
        }
    };

    class Trajectory {
    public:
        using Scalar = Biped::rcg::Scalar;
        using Nodes = std::vector<Node<Scalar>>;
        using NodesRef = std::shared_ptr<Nodes>;
        using ModelRef = std::shared_ptr<Model>;

        Trajectory(
            ModelRef model,
            NodesRef base_angular_nodes,
            NodesRef base_linear_nodes,
            NodesRef joint_state_nodes,
            NodesRef contact_force_nodes,
            NodesRef control_nodes,
            const Scalar& dt)
            : model{std::move(model)},
              base_angular(std::move(base_angular_nodes), dt, 3),
              base_linear(std::move(base_linear_nodes), dt, 3),
              joint_state(std::move(joint_state_nodes), dt, joint_space_dim),
              contact_force(std::move(contact_force_nodes), dt, 3),
              control(std::move(control_nodes), dt, joint_space_dim) {}

    private:
        Spline<Scalar, RotationAdaptor<Scalar>> base_angular;
        Spline<Scalar, Polynomial<Scalar>> base_linear;
        Spline<Scalar, Polynomial<Scalar>> joint_state;
        Spline<Scalar, Polynomial<Scalar>> contact_force;
        Spline<Scalar, Polynomial<Scalar>> control;

        ModelRef model;
    };

}// namespace to

#endif//TO_HZD_MODEL_H
