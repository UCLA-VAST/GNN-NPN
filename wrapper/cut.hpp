#pragma once

#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/reconv_cut.hpp>
#include <mockturtle/io/write_aiger.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/utils/node_map.hpp>

#include "aig.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static mockturtle::aig_network make_subproblem(
  mockturtle::aig_network const& aig,
  mockturtle::aig_network::node const& pivot,
  int cut_size
)
{
  using namespace mockturtle;
  using node   = aig_network::node;
  using signal = aig_network::signal;

  auto s   = aig.make_signal( pivot );
  auto res = reconvergence_driven_cut<aig_network, true, false>(
    aig, s, reconvergence_driven_cut_parameters{ (uint64_t)cut_size } );

  auto& leaves = res.first;
  auto& nodes  = res.second;

  if ( (int)leaves.size() != cut_size )
    return {};  // sentinel: empty network

  std::sort( std::begin( nodes ), std::end( nodes ) );

  aig_network src;
  unordered_node_map<signal, aig_network> old_to_new( aig );
  old_to_new[aig.get_node( aig.get_constant( false ) )] = src.get_constant( false );
  for ( auto const& n : leaves )
    old_to_new[n] = src.create_pi();

  std::vector<signal> children;
  for ( node const& n : nodes )
  {
    if ( old_to_new.has( n ) ) continue;
    children.clear();
    aig.foreach_fanin( n, [&]( auto child, auto ) {
      const auto f = old_to_new[aig.get_node( child )];
      children.push_back( aig.is_complemented( child )
        ? src.create_not( f ) : f );
    } );
    old_to_new[n] = src.create_and( children[0], children[1] );
  }
  src.create_po( old_to_new[pivot] );

  aig_network subproblem;
  std::vector<signal> dest_pis;
  src.foreach_pi( [&]( auto const& ) {
    dest_pis.push_back( subproblem.create_pi() );
  } );
  auto pos = cleanup_dangling( src, subproblem, dest_pis.begin(), dest_pis.end() );
  for ( auto const& po : pos )
    subproblem.create_po( po );

  if ( (int)subproblem.num_pis() != cut_size || !all_pis_have_fanout( subproblem ) )
    return {};  // sentinel: empty network

  return subproblem;
}

uint32_t cut_aig(
  std::string const& aig_path,
  std::string const& out_dir,
  int cut_size
)
{
  using namespace mockturtle;

  fs::path out_aig{ out_dir };
  fs::create_directories( out_aig );

  auto aig = load_aig( aig_path );

  uint32_t n_subcircuits = 0;
  aig.foreach_gate( [&]( auto const& pivot ) {
    if ( aig.fanout_size( pivot ) == 0u ) return;

    auto subproblem = make_subproblem( aig, pivot, cut_size );
    if ( subproblem.num_pis() == 0u ) return;  // invalid cut, skipped

    write_aiger( subproblem, ( out_aig / ( std::to_string( pivot ) + ".aig" ) ).string() );
    n_subcircuits += 1;
  } );

  return n_subcircuits;
}
