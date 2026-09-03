#pragma once

#include <stdexcept>
#include <string>

#include <kitty/constructors.hpp>
#include <kitty/dynamic_truth_table.hpp>

inline kitty::dynamic_truth_table make_tt(
  const std::string& tt_hex_string
)
{
  const auto hex_size = tt_hex_string.size();
  if ( hex_size == 0u || ( hex_size & ( hex_size - 1u ) ) != 0u )
  {
    throw std::invalid_argument( "truth table hex string length must be a non-zero power of two" );
  }

  const auto num_vars = 2u + static_cast<unsigned>( __builtin_ctzll( hex_size ) );
  kitty::dynamic_truth_table tt( num_vars );
  kitty::create_from_hex_string( tt, tt_hex_string );
  return tt;
}
