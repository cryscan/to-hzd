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
        biped::rcg::HomogeneousTransforms homogeneous_transforms;
        biped::rcg::MotionTransforms motion_transforms;
        biped::rcg::ForceTransforms force_transforms;
        biped::rcg::Jacobians jacobians;
        biped::rcg::InertiaProperties inertia_properties;

        biped::rcg::InverseDynamics inverse_dynamics;
        biped::rcg::JSIM jsim;

        Model()
            : inverse_dynamics(inertia_properties, motion_transforms),
              jsim(inertia_properties, force_transforms) {
        }
    };
    struct State {
        biped::rcg::JointState q, qd, qdd;
        Eigen::Quaternion<biped::rcg::Scalar> r;
        Eigen::Vector3<biped::rcg::Scalar> rd, rdd;
        Eigen::Vector3<biped::rcg::Scalar> p, pd, pdd;
    };

}// namespace to

#endif//TO_HZD_MODEL_H
