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

namespace to {
    static constexpr int joint_space_dim = Biped::rcg::JointSpaceDimension;
    static constexpr int pose_state_dim = 7 + joint_space_dim;
    static constexpr int state_dim = 6 + joint_space_dim;

    struct State {
        Biped::rcg::JointState q, qd, qdd;
        Eigen::Quaternion<Biped::rcg::Scalar> r;
        Eigen::Vector3<Biped::rcg::Scalar> rd, rdd;
        Eigen::Vector3<Biped::rcg::Scalar> p, pd, pdd;

        [[nodiscard]] Eigen::Vector<Biped::rcg::Scalar, state_dim> velocity() const {
            Eigen::Vector<Biped::rcg::Scalar, state_dim> xd;
            xd << rd, pd, qd;
            return xd;
        }

        [[nodiscard]] Eigen::Vector<Biped::rcg::Scalar, state_dim> acceleration() const {
            Eigen::Vector<Biped::rcg::Scalar, state_dim> xdd;
            xdd << rdd, pdd, qdd;
            return xdd;
        }
    };

    class Model {
    public:
        using Scalar = Biped::rcg::Scalar;

        Model()
            : inverse_dynamics(inertia_properties, motion_transforms),
              jsim(inertia_properties, force_transforms) {
        }

        enum class Side {
            LEFT,
            RIGHT
        };

        Eigen::Matrix<Scalar, 6, state_dim> foot_jacobian(const State& state, Side side) {
            Eigen::Matrix<Scalar, 6, joint_space_dim> joint_space_jacobian = Eigen::Matrix<Scalar, 6, joint_space_dim>::Zero();
            Eigen::Vector4<Scalar> p;
            switch (side) {
                case Side::LEFT:
                    joint_space_jacobian.leftCols<3>() << jacobians.fr_trunk_J_L_foot(state.q);
                    p << homogeneous_transforms.fr_trunk_X_L_foot * Eigen::Vector4<Scalar>{0, 0, 0, 1};
                    break;
                case Side::RIGHT:
                    joint_space_jacobian.rightCols<3>() << jacobians.fr_trunk_J_R_foot(state.q);
                    p << homogeneous_transforms.fr_trunk_X_R_foot * Eigen::Vector4<Scalar>{0, 0, 0, 1};
                    break;
            }

            Eigen::Matrix3<Scalar> px;
            px << 0, -p(2), p(1),
                p(2), 0, -p(0),
                -p(1), p(0), 0;

            Eigen::Matrix<Scalar, 6, state_dim> jacobian = Eigen::Matrix<Scalar, 6, state_dim>::Zero();
            Eigen::Matrix3<Scalar> r = state.r.toRotationMatrix();
            jacobian.block<3, 3>(0, 0) = r;                                 // Base angular -> EE angular
            jacobian.block<3, 3>(3, 0) = -r * px;                           // Base angular -> EE linear
            jacobian.block<3, 3>(0, 3) = Eigen::Matrix3<Scalar>::Zero();    // Base linear  -> EE angular
            jacobian.block<3, 3>(3, 3) = Eigen::Matrix3<Scalar>::Identity();// Base linear  -> EE linear
            jacobian.block<3, joint_space_dim>(0, 6) = r * joint_space_jacobian.topRows<3>();
            jacobian.block<3, joint_space_dim>(3, 6) = r * joint_space_jacobian.bottomRows<3>();

            return jacobian;
        }

    private:
        Biped::rcg::HomogeneousTransforms homogeneous_transforms;
        Biped::rcg::MotionTransforms motion_transforms;
        Biped::rcg::ForceTransforms force_transforms;
        Biped::rcg::Jacobians jacobians;
        Biped::rcg::InertiaProperties inertia_properties;

        Biped::rcg::InverseDynamics inverse_dynamics;
        Biped::rcg::JSIM jsim;
    };

    class Trajectory {
    public:
        using Scalar = Biped::rcg::Scalar;

    private:
        Spline<Scalar, RotationAdaptor<Scalar>> base_angular;
        Spline<Scalar, Polynomial<Scalar>> base_linear;
        Spline<Scalar, Polynomial<Scalar>> joint_state;
    };

}// namespace to

#endif//TO_HZD_MODEL_H
