//
// Created by cryscan on 2/2/23.
//

#ifndef TO_HZD_UTILS_H
#define TO_HZD_UTILS_H

#include <Eigen/Eigen>
#include <cppad/cppad.hpp>

namespace to {
    // Exponential map that maps R^3 to S^3.
    // Reference: https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf
    template<typename Scalar>
    Eigen::Quaternion<Scalar> exp(const Eigen::Vector3<Scalar> v) {
        Scalar _1_2{0.5}, _1{1.0}, _8{8.0}, _48{48.0}, _384{384.0};
        Scalar eps{0.00012207};

        Scalar t2 = v.squaredNorm();
        Scalar t4 = t2 * t2;
        Scalar t = CppAD::sqrt(t2);
        Scalar st = CppAD::sin(_1_2 * t);
        Scalar ct = CppAD::CondExpLe(t, eps, _1 - t2 / _8 + t4 / _384, CppAD::cos(_1_2 * t));
        Scalar sc = CppAD::CondExpLe(t, eps, _1_2 + t2 / _48, st / t);

        Eigen::Vector3<Scalar> sv = sc * v;
        return {ct, sv(0), sv(1), sv(2)};
    }

    // Raising a vector to a skew-symmetric matrix.
    template<typename Scalar, typename Vector>
    Eigen::Matrix3<Scalar> hat(const Vector& v) {
        Eigen::Matrix3<Scalar> m;
        m << 0, -v(2), v(1),
            v(2), 0, -v(0),
            -v(1), v(0), 0;
        return m;
    }
}// namespace to

#endif//TO_HZD_UTILS_H
