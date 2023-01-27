#include <iostream>

#include "biped/declarations.h"
#include "biped/transforms.h"
#include "biped/jacobians.h"
#include "biped/inertia_properties.h"
#include "biped/inverse_dynamics.h"
#include "biped/jsim.h"

using namespace biped;

int main() {
    rcg::JointState q{0.0, M_PI_4, -M_PI_2, 0.0, M_PI_4, -M_PI_2};
    rcg::JointState qd, qdd, tau;
    rcg::Acceleration a;
    rcg::Vector3 p;
    rcg::Matrix<4, 1> r;

    rcg::Acceleration g{0.0, 0.0, 0.0, 0.0, 0.0, -rcg::g};
    rcg::Velocity v{};

    rcg::MotionTransforms motion_transforms;
    rcg::HomogeneousTransforms homogeneous_transforms;
    rcg::ForceTransforms force_transforms;
    rcg::InertiaProperties inertia_properties;
    rcg::InverseDynamics inverse_dynamics(inertia_properties, motion_transforms);
    rcg::JSIM jsim(inertia_properties, force_transforms);
    rcg::Jacobians jacobians;

    inverse_dynamics.id(tau, a, g, v, q, qd, qdd);
    std::cout << tau.transpose() << "\n";
    std::cout << a.transpose() << "\n";
    std::cout << "\n";

    std::cout << jsim(q) << "\n\n";

    jsim.computeL();
    jsim.computeInverse();
    std::cout << jsim.getInverse() << "\n\n";

    std::cout << jacobians.fr_trunk_J_L_foot(q) << "\n\n";

    return 0;
}
