#pragma once

#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/simulation.hpp>
#include <mockturtle/generators/random_network.hpp>
#include <mockturtle/io/aiger_reader.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/views/topo_view.hpp>
#include <lorina/aiger.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

mockturtle::aig_network rewrite( mockturtle::aig_network const& in );

inline bool all_pis_have_fanout(
  mockturtle::aig_network const& ntk
)
{
  for ( uint32_t i = 0; i < ntk.num_pis(); ++i )
  {
    if ( ntk.fanout_size( ntk.pi_at( i ) ) == 0u )
      return false;
  }
  return true;
}

inline mockturtle::aig_network load_aig(
  const std::string& aig_path
)
{
  mockturtle::aig_network aig;
  const auto rc = lorina::read_aiger( aig_path, mockturtle::aiger_reader( aig ) );
  if ( rc != lorina::return_code::success )
  {
    throw std::runtime_error( "failed to read AIGER file: " + aig_path );
  }
  return aig;
}

inline std::string simulate_aig(
  const mockturtle::aig_network& aig
)
{
  mockturtle::topo_view topo{ aig };
  mockturtle::default_simulator<kitty::dynamic_truth_table> simulator( aig.num_pis() );
  kitty::dynamic_truth_table res = mockturtle::simulate<kitty::dynamic_truth_table>(topo, simulator)[0];
  return kitty::to_hex(res);
}

inline std::string aig_to_tt(
  const std::string& aig_path
)
{
  auto aig = load_aig( aig_path );
  return simulate_aig( aig );
}

// Gate-type indices matching deepgate's convention: PI=0, AND=1, NOT=2
static constexpr int GATE_PI  = 0;
static constexpr int GATE_AND = 1;
static constexpr int GATE_NOT = 2;

using XData     = std::vector<std::vector<int>>;
using EdgeIndex = std::vector<std::vector<int>>;

// Convert a mockturtle aig_network to deepgate's (x_data, edge_index) format.
//   x_data  : [[node_id, type], ...] where type PI=0, AND=1, NOT=2.
//             PIs first (foreach_pi order), then AND gates (foreach_gate order),
//             then NOT nodes appended on demand for complemented edges.
//   edge_index: [[src, dst], ...] directed signal-flow edges.
inline std::tuple<XData, EdgeIndex> aig_to_xdata(
  const mockturtle::aig_network& aig
)
{
  using node_t   = mockturtle::aig_network::node;
  using signal_t = mockturtle::aig_network::signal;

  XData     x_data;
  EdgeIndex edge_index;

  std::unordered_map<uint32_t, int> node_idx; // mockturtle node id → dense index

  // 1. PIs
  aig.foreach_pi( [&]( node_t n ) {
    int idx = static_cast<int>( x_data.size() );
    node_idx[n] = idx;
    x_data.push_back( { idx, GATE_PI } );
  } );

  // 2. AND gates
  aig.foreach_gate( [&]( node_t n ) {
    int idx = static_cast<int>( x_data.size() );
    node_idx[n] = idx;
    x_data.push_back( { idx, GATE_AND } );
  } );

  // has_not[dense_idx] = index of its NOT node, or -1 if none yet
  std::vector<int> has_not( x_data.size(), -1 );

  auto get_not_node = [&]( int src_idx ) -> int {
    if ( has_not[src_idx] != -1 )
      return has_not[src_idx];
    int not_idx = static_cast<int>( x_data.size() );
    x_data.push_back( { not_idx, GATE_NOT } );
    has_not.push_back( -1 );
    edge_index.push_back( { src_idx, not_idx } );
    has_not[src_idx] = not_idx;
    return not_idx;
  };

  // 3. Edges for AND gates
  aig.foreach_gate( [&]( node_t n ) {
    int and_idx = node_idx.at( n );
    aig.foreach_fanin( n, [&]( signal_t fi ) {
      node_t fi_node = aig.get_node( fi );
      if ( aig.is_constant( fi_node ) )
        throw std::runtime_error( "aig_to_xdata: gate has a constant fanin" );
      int src = node_idx.at( fi_node );
      if ( aig.is_complemented( fi ) )
        src = get_not_node( src );
      edge_index.push_back( { src, and_idx } );
    } );
  } );

  // 4. PO complement bit
  aig.foreach_po( [&]( signal_t po ) {
    node_t po_node = aig.get_node( po );
    if ( aig.is_constant( po_node ) ) return;  // skip constant POs
    int src = node_idx.at( po_node );
    if ( aig.is_complemented( po ) )
      get_not_node( src );
  } );

  return { x_data, edge_index };
}

namespace detail
{
inline std::optional<mockturtle::aig_network> try_random_single_po_aig(
  uint32_t num_pis,
  uint32_t num_gates,
  uint64_t seed
)
{
  mockturtle::random_network_generator_params_size ps;
  ps.num_pis   = num_pis;
  ps.num_gates = num_gates;
  ps.seed      = seed;
  ps.num_networks_per_configuration = 1u;

  auto gen = mockturtle::random_aig_generator( ps );
  mockturtle::aig_network raw = gen.generate();

  // Use only the last AND gate as the single PO.
  mockturtle::aig_network::node last_gate{};
  bool has_gate = false;
  raw.foreach_gate( [&]( mockturtle::aig_network::node n ) {
    last_gate = n;
    has_gate  = true;
  } );
  if ( !has_gate ) return std::nullopt;

  // Build a fresh network with a single PO at last_gate.
  mockturtle::aig_network aig;
  // Copy PIs.
  for ( uint32_t i = 0; i < raw.num_pis(); ++i )
    aig.create_pi();
  // Copy all gates using topo order.
  std::unordered_map<uint32_t, mockturtle::aig_network::signal> old_to_new;
  old_to_new[raw.get_node( raw.get_constant( false ) )] = aig.get_constant( false );
  uint32_t pi_idx = 0;
  raw.foreach_pi( [&]( mockturtle::aig_network::node n ) {
    old_to_new[n] = aig.make_signal( aig.pi_at( pi_idx++ ) );
  } );
  mockturtle::topo_view topo{ raw };
  topo.foreach_node( [&]( mockturtle::aig_network::node n ) {
    if ( raw.is_constant( n ) || raw.is_pi( n ) ) return;
    mockturtle::aig_network::signal f0, f1;
    bool got = false;
    raw.foreach_fanin( n, [&]( mockturtle::aig_network::signal fi ) {
      auto s = old_to_new.at( raw.get_node( fi ) );
      if ( raw.is_complemented( fi ) ) s = !s;
      if ( !got ) { f0 = s; got = true; } else { f1 = s; }
    } );
    old_to_new[n] = aig.create_and( f0, f1 );
  } );
  aig.create_po( old_to_new.at( last_gate ) );

  aig = mockturtle::cleanup_dangling( aig );
  if ( aig.num_pis() != num_pis || !all_pis_have_fanout( aig ) )
    return std::nullopt;

  return aig;
}

struct random_aig_sampler
{
  random_aig_sampler( uint32_t num_pis, double mean_gates, double std_gates, uint64_t seed )
    : num_pis_( num_pis ),
      mean_gates_( mean_gates ),
      use_rejection_( mean_gates > 0.0 && std_gates > 0.0 ),
      inv_2var_( use_rejection_ ? 1.0 / ( 2.0 * std_gates * std_gates ) : 0.0 ),
      rng_( seed == 0 ? std::random_device{}() : seed ),
      gates_dist_( 1u, proposal_ceiling( mean_gates, std_gates ) ),
      accept_dist_( 0.0, 1.0 )
  {}

  // One proposal draw; nullopt if it fails the structural checks.
  std::optional<mockturtle::aig_network> next_candidate()
  {
    uint32_t num_gates    = gates_dist_( rng_ );
    uint64_t attempt_seed = rng_();
    return try_random_single_po_aig( num_pis_, num_gates, attempt_seed );
  }

  // Accept with probability N(num_gates)/N(mean) = exp(-(n-mean)^2/(2*std^2))
  bool accept( uint32_t num_gates )
  {
    if ( !use_rejection_ ) return true;
    double diff = static_cast<double>( num_gates ) - mean_gates_;
    return accept_dist_( rng_ ) <= std::exp( -diff * diff * inv_2var_ );
  }

private:
  static uint32_t proposal_ceiling( double mean_gates, double std_gates )
  {
    if ( mean_gates <= 0.0 ) return 64u;
    return static_cast<uint32_t>( std::max( 1.0, std::ceil( ( mean_gates + std_gates * 4.0 ) * 4.0 ) ) );
  }

  uint32_t num_pis_;
  double   mean_gates_;
  bool     use_rejection_;
  double   inv_2var_;
  std::mt19937_64 rng_;
  std::uniform_int_distribution<uint32_t> gates_dist_;
  std::uniform_real_distribution<double>  accept_dist_;
};

}

inline std::tuple<XData, EdgeIndex, std::string> generate_random_aig(
  uint32_t num_pis,
  double mean_nodes = 0.0,
  double std_nodes = 0.0,
  uint64_t seed = 0
)
{
  detail::random_aig_sampler sampler( num_pis, mean_nodes, std_nodes, seed );
  while ( true )
  {
    auto aig = sampler.next_candidate();
    if ( !aig ) continue;
    if ( !sampler.accept( aig->num_gates() ) ) continue;

    auto [x_data, edge_index] = aig_to_xdata( *aig );
    return { x_data, edge_index, simulate_aig( *aig ) };
  }
}

inline std::tuple<XData, EdgeIndex, std::string> generate_random_aig_rewriting(
  uint32_t num_pis,
  double mean_nodes = 0.0,
  double std_nodes = 0.0,
  uint64_t seed = 0
)
{
  detail::random_aig_sampler sampler( num_pis, mean_nodes, std_nodes, seed );
  while ( true )
  {
    auto aig = sampler.next_candidate();
    if ( !aig ) continue;
    if ( !sampler.accept( aig->num_gates() ) ) continue;

    auto rw_aig = rewrite( *aig );
    if ( rw_aig.num_gates() == 0 ) continue;

    auto [x_data, edge_index] = aig_to_xdata( rw_aig );
    return { x_data, edge_index, simulate_aig( rw_aig ) };
  }
}
