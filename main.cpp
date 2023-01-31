#include <iostream>

#include "model.h"
#include "spline.h"

int main() {
    using namespace biped::rcg;

    Scalar t{7.5};

    {
        to::Polynomial<Scalar> polynomial(10.0, 4);

        to::Node<Scalar> head{Eigen::Vector4<Scalar>{1.0, 0.0, 0.0, 0.0}, Eigen::Vector4<Scalar>{0.0, 0.0, 0.0, 0.0}};
        to::Node<Scalar> tail{Eigen::Vector4<Scalar>{0.0, 1.0, 0.0, 0.0}, Eigen::Vector4<Scalar>{0.0, 0.0, 0.0, 0.0}};

        polynomial.update_nodes(head, tail);

        std::cout << polynomial.eval(t, 0).transpose() << '\n';
        std::cout << polynomial.eval(t, 1).transpose() << '\n';
        std::cout << polynomial.eval(t, 2).transpose() << "\n\n";
    }

    {
        to::Node<Scalar> head{Eigen::Vector3<Scalar>{M_PI_2, 0.0, 0.0}, Eigen::Vector3<Scalar>::Zero()};
        to::Node<Scalar> tail{Eigen::Vector3<Scalar>{0.0, M_PI_2, 0.0}, Eigen::Vector3<Scalar>::Zero()};

        to::RotationAdaptor<Scalar> rotation_adaptor(10.0);
        rotation_adaptor.update_nodes(head, tail);

        std::cout << rotation_adaptor.eval(t, 0).transpose() << '\n';
        std::cout << rotation_adaptor.eval(t, 1).transpose() << '\n';
        std::cout << rotation_adaptor.eval(t, 2).transpose() << '\n';

        std::cout << rotation_adaptor.eval_qd(t).transpose() << "\n\n";
    }

    {
        std::vector<to::Node<Scalar>> nodes{
            {Eigen::Vector3<Scalar>{0.0, 0.0, 0.0}, Eigen::Vector3<Scalar>::Zero()},
            {Eigen::Vector3<Scalar>{M_PI_2, 0.0, 0.0}, Eigen::Vector3<Scalar>{0.0, 0.0, 1.0}},
            {Eigen::Vector3<Scalar>{0.0, M_PI_2, 0.0}, Eigen::Vector3<Scalar>{1.0, 0.0, 0.0}},
            {Eigen::Vector3<Scalar>{0.0, 0.0, M_PI_2}, Eigen::Vector3<Scalar>::Zero()},
        };

        to::Spline<Scalar, to::RotationAdaptor<Scalar>> spline(nodes, 0.1, 3);

        t = Scalar{0.25};

        std::cout << spline.polynomial_id(t) << '\n';
        std::cout << spline.eval(t, 0).transpose() << '\n';
        std::cout << spline.eval(t, 1).transpose() << '\n';
        std::cout << spline.eval(t, 2).transpose() << "\n\n";
    }

    return 0;
}
