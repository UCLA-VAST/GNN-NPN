#pragma once

#include <kitty/npn.hpp>
#include <kitty/print.hpp>

#include "tt.hpp"

#include <cstdint>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

inline std::tuple<std::vector<bool>, bool> translate_kitty_neg(
  uint32_t neg,
  std::size_t num_vars
)
{
  std::vector<bool> in_neg( num_vars, false );
  for ( uint32_t i = 0; i < num_vars; ++i )
  {
    in_neg[i] = ( neg & ( 1u << i ) ) != 0u;
  }
  bool out_neg = ( neg & ( 1u << num_vars ) ) != 0u;
  return std::make_tuple( in_neg, out_neg );
}

inline std::vector<uint8_t> make_identity_perm(
  std::size_t num_vars
)
{
  std::vector<uint8_t> perm( num_vars );
  std::iota( perm.begin(), perm.end(), 0u );
  return perm;
}

inline std::tuple<std::string, std::vector<bool>, std::vector<uint8_t>, bool> exact_npn(
  const std::string& tt_hex_string
)
{
  auto tt = make_tt( tt_hex_string );
  auto [tt_new, neg, in_perm] = kitty::exact_npn_canonization( tt );
  auto [in_neg, out_neg] = translate_kitty_neg( neg, tt.num_vars() );
  return std::make_tuple(kitty::to_hex( tt_new ), in_neg, in_perm, out_neg);
}

inline std::tuple<std::string, std::vector<bool>, std::vector<uint8_t>, bool> exact_n(
  const std::string& tt_hex_string
)
{
  auto tt = make_tt( tt_hex_string );
  auto [tt_new, neg] = kitty::exact_n_canonization( tt );
  auto [in_neg, out_neg] = translate_kitty_neg( neg, tt.num_vars() );
  return std::make_tuple(kitty::to_hex( tt_new ), in_neg, make_identity_perm( tt.num_vars() ), out_neg);
}

inline std::tuple<std::string, std::vector<bool>, std::vector<uint8_t>, bool> exact_p(
  const std::string& tt_hex_string
)
{
  auto tt = make_tt( tt_hex_string );
  auto [tt_new, neg, in_perm] = kitty::exact_p_canonization( tt );
  auto [in_neg, out_neg] = translate_kitty_neg( neg, tt.num_vars() );
  return std::make_tuple(kitty::to_hex( tt_new ), in_neg, in_perm, out_neg);
}

inline std::tuple<std::string, std::vector<bool>, std::vector<uint8_t>, bool> flip_swap_npn(
  const std::string& tt_hex_string
)
{
  auto tt = make_tt( tt_hex_string );
  auto [tt_new, neg, in_perm] = kitty::flip_swap_npn_canonization( tt );
  auto [in_neg, out_neg] = translate_kitty_neg( neg, tt.num_vars() );
  return std::make_tuple(kitty::to_hex( tt_new ), in_neg, in_perm, out_neg);
}

inline std::tuple<std::string, std::vector<bool>, std::vector<uint8_t>, bool> sifting_npn(
  const std::string& tt_hex_string
)
{
  auto tt = make_tt( tt_hex_string );
  auto [tt_new, neg, in_perm] = kitty::sifting_npn_canonization( tt );
  auto [in_neg, out_neg] = translate_kitty_neg( neg, tt.num_vars() );
  return std::make_tuple(kitty::to_hex( tt_new ), in_neg, in_perm, out_neg);
}
