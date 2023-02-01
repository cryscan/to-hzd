//
// Created by cryscan on 1/30/23.
//

#ifndef TO_HZD_FORWARD_H
#define TO_HZD_FORWARD_H

namespace to {
    template<typename Scalar>
    class Node;

    template<typename Scalar>
    class Polynomial;

    template<typename Scalar>
    class RotationAdaptor;

    template<typename Polynomial, typename Scalar>
    concept PolynomialType = requires(Polynomial& p, const Node<Scalar>& node, const Scalar& t, int n) {
                                 { p.D } -> std::convertible_to<int>;
                                 { p.update_nodes(node, node) };
                                 { p.eval(t, n) } -> std::convertible_to<Eigen::VectorX<Scalar>>;
                             };

    template<typename Scalar, PolynomialType<Scalar> Polynomial>
    class Spline;


    class Model;
    class State;
    class Trajectory;
}// namespace to

#endif//TO_HZD_FORWARD_H
