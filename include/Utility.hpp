#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <type_traits>

namespace LinearAlgebra {

template<typename... Args>
constexpr void assertTypesAreArithmetic() {
    static_assert((std::is_arithmetic<Args>::value && ...), 
                  "Error: non-arithmetic types detected.");
}
} // namespace LinearAlgebra
#endif // UTILITY_HPP