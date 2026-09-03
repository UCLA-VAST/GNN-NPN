#pragma once

#include <kitty/npn.hpp>
#include <kitty/print.hpp>

#include "tt.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

inline uint32_t translate_python_neg(
  const std::vector<bool>& in_neg,
  bool out_neg
)
{
  uint32_t neg = 0u;
  for ( std::size_t i = 0; i < in_neg.size(); ++i )
  {
    if ( in_neg[i] )
    {
      neg |= 1u << i;
    }
  }
  if ( out_neg )
  {
    neg |= 1u << in_neg.size();
  }
  return neg;
}

inline void validate_npn_transform(
  std::size_t num_vars,
  const std::vector<bool>& in_neg,
  const std::vector<uint8_t>& perm
)
{
  if ( in_neg.size() != num_vars )
  {
    throw std::invalid_argument( "in_neg size must match the number of variables" );
  }
  if ( perm.size() != num_vars )
  {
    throw std::invalid_argument( "perm size must match the number of variables" );
  }

  std::vector<bool> seen( num_vars, false );
  for ( auto value : perm )
  {
    if ( value >= num_vars )
    {
      throw std::invalid_argument( "perm contains an out-of-range index" );
    }
    if ( seen[value] )
    {
      throw std::invalid_argument( "perm must be a valid permutation" );
    }
    seen[value] = true;
  }
}

inline std::string apply_to_tt(
    const std::string& tt_hex_string,
    const std::vector<bool>& in_neg,
    const std::vector<uint8_t>& perm,
    bool out_neg
)
{
  auto tt = make_tt( tt_hex_string );
  validate_npn_transform( tt.num_vars(), in_neg, perm );
  auto neg = translate_python_neg( in_neg, out_neg );
  auto tt_transformed = kitty::apply_npn_transformation( tt, neg, perm );
  return kitty::to_hex( tt_transformed );
}
