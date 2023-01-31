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
    struct Model {
        Biped::rcg::HomogeneousTransforms homogeneous_transforms;
        Biped::rcg::MotionTransforms motion_transforms;
        Biped::rcg::ForceTransforms force_transforms;
        Biped::rcg::Jacobians jacobians;
        Biped::rcg::InertiaProperties inertia_properties;

        Biped::rcg::InverseDynamics inverse_dynamics;
        Biped::rcg::JSIM jsim;

        Model()
            : inverse_dynamics(inertia_properties, motion_transforms),
              jsim(inertia_properties, force_transforms) {
        }
    };

    struct State {
        Biped::rcg::JointState q, qd, qdd;
        Eigen::Quaternion<Biped::rcg::Scalar> r;
        Eigen::Vector3<Biped::rcg::Scalar> rd, rdd;
        Eigen::Vector3<Biped::rcg::Scalar> p, pd, pdd;
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
