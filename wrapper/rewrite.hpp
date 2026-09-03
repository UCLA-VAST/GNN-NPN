#pragma once

#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/experimental/cost_generic_resub.hpp>
#include <mockturtle/io/write_aiger.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/utils/recursive_cost_functions.hpp>

#include "aig.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

struct aig_ambiguity_cost
: mockturtle::recursive_cost_functions<mockturtle::aig_network>
{
	using Ntk = mockturtle::aig_network;
	using node = mockturtle::node<Ntk>;
	using signal = mockturtle::signal<Ntk>;
	using context_t = uint32_t; // level(n)

	static bool match(Ntk const& ntk, node n, node l, bool inv_edge) {
		return ntk.is_and(n) && ntk.is_and(l) && !inv_edge;
	}

	context_t operator()( Ntk const& ntk,
	                      node const& n,
	                      std::vector<context_t> const& fanin_ctx = {} ) const override
	{
		if ( ntk.is_constant(n) || ntk.is_pi(n) ) return 0u;

		// level(n) = min_l { level(l)       if match(n,l)
		//                     level(l) + 1   otherwise }
		uint32_t best = std::numeric_limits<uint32_t>::max();
		uint32_t i = 0;
		ntk.foreach_fanin( n, [&]( signal const& f ){
			const auto l = ntk.get_node(f);
			const auto lev_l = fanin_ctx.empty() ? 0u : fanin_ctx[i++];
			const auto cand = match(ntk, n, l, ntk.is_complemented(f)) ? lev_l : (lev_l + 1);
			best = std::min(best, cand);
		});
		return (best == std::numeric_limits<uint32_t>::max()) ? 0u : best;
	}

	void operator()( Ntk const& ntk,
	                 node const& n,
	                 uint32_t& total_cost,
	                 context_t const ctx ) const override
	{
		if ( ntk.is_constant(n) || ntk.is_pi(n) ) return;
		total_cost = std::max(total_cost, ctx); // Cost(C) = max_n level(n)
	}
};

mockturtle::aig_network rewrite(
  mockturtle::aig_network const& in
)
{
	auto ntk = in.clone();
	mockturtle::experimental::cost_generic_resub_params ps;
	ps.wps.max_pis = std::min( ntk.num_pis(), 4u );
	mockturtle::experimental::cost_generic_resub_stats st;
	mockturtle::experimental::cost_generic_resub( ntk, aig_ambiguity_cost{}, ps, &st );
	ntk = mockturtle::cleanup_dangling( ntk, /*remove_dangling_PIs=*/false );
	if ( ntk.num_pis() != in.num_pis() || !all_pis_have_fanout( ntk ) )
		return in;
	return ntk;
}

inline void rewrite_aig(
  std::string const& in_path,
  std::string const& out_path
)
{
	auto aig = load_aig( in_path );
	mockturtle::write_aiger( rewrite( aig ), out_path );
}
