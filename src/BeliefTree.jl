using POMDPs
include("Bounds.jl")
# A belief tree, the root node stores the initial belief

mutable struct BeliefTreeNode
    _state_particles::Vector{Any}
    _child_nodes::Dict{Pair{Any, Any}, BeliefTreeNode}
    _best_action::Any
    _R_a::Dict{Any, Float64} # a map from actions to expected instant reward, not sure it's useful or not
    _upper_bound::Float64 
    _lower_bound::Float64
    _fsc_node_index::Int64 # link to a FSC node index, default is -1
end



"""
Create new belief tree node with U and L initilizations
"""
function CreateBelieTreefNode(b_tree_node_parent::BeliefTreeNode, a, o, b_new, Q_learning_policy::Qlearning, pomdp)
    a_best_new_belief, U = EvaluateUpperBound(b_new, Q_learning_policy)
    new_tree_node = BeliefTreeNode(b_new, Dict{Pair{Any, Any}, BeliefTreeNode}(), a_best_new_belief, Dict{Any, Float64}(), U, FindRLower(pomdp, b_new, actions(pomdp)), -1)
    b_tree_node_parent._child_nodes[Pair(a, o)] = new_tree_node
end



"""
Sample Beliefs from a belief tree with heuristics
"""
function SampleBeliefs(root::BeliefTreeNode, s::Any, depth::Int64, L::Int64, nb_sim::Int64, pomdp, Q_learning_policy::Qlearning, b_list_out)
    # Sample beliefs within considered depth
    if depth < L # very naive condition!!! should improve it later
        # choose the best action
        a = root._best_action
        # should choose an observation that maximize (U - L) for every b_a_o
        # currently just choose the received observation
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
        # check if this ao edge exist
        if !haskey(root._child_nodes, Pair(a, o))
            # Belief update and add the corresponding beliefs
            o, next_beliefs = BeliefUpdate(root, a, nb_sim, pomdp) 
            for (o_temp, b_next) in next_beliefs
                CreateBelieTreefNode(root, a, o_temp, b_next, Q_learning_policy, pomdp)
            end
        end

        # recursive sampling
        push!(b_list_out, root)
        SampleBeliefs(root._child_nodes[Pair(a, o)], sp, depth + 1, L, nb_sim, pomdp, Q_learning_policy, b_list_out)
        # update the U value??? U will be updated in MC-backup ???
    end
end


"""
Choose the observation that contribute the most to the (U-L) gap at root node
"""
function ChooseObservation(node::BeliefTreeNode, a_best::Any)
    o_selected = rand(obs_space)
    bool_has_childs = false

    # if node has childs with action a_best
    for o in obs_space
        ao_edge = Pair(a_best, o)
        if haskey(root._child_nodes[ao_edge])
            bool_has_childs = true
            U_child = root._child_nodes[ao_edge]._upper_bound
            L_child = root._child_nodes[ao_edge]._lower_bound
            gap_child = U_child - L_child
            if gap_child > largest_gap
                largest_gap = gap_child
                o_selected = o
            end            
        end
    end 

    return bool_has_childs, o_selected
end

# Update Beliefs with MC sampling
function BeliefUpdate(node::BeliefTreeNode, a, nb_sim::Int64, pomdp)
    next_beliefs = Dict{Any, Any}() # a map from observations to beliefs
    observation_counts = Dict{Any, Int64}() # a map that stores observation counts

    # do simulations to gather particles
    sum_r = 0.0
    for i in 1:nb_sim
        s = rand(node._state_particles)
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
        sum_r += r
        if haskey(next_beliefs, o)
            push!(next_beliefs[o], sp)
            observation_counts[o] += 1
        else
            next_beliefs[o] = [sp]
            observation_counts[o] = 1
        end
    end

    node._R_a[a] = sum_r / nb_sim

    # find which observation is the most probable one
    o_selected = first(observation_counts).first
    o_selected_counts = first(observation_counts).second
    for (k,v) in observation_counts
        if v > o_selected_counts
            o_selected_counts = v
            o_selected = k
        end
    end


    # return the most probable observation and next beliefs (a map from observations to beliefs)
    return o_selected, next_beliefs 
end


function FindRLower(pomdp, b0, action_space)
	action_min_r = Dict{Any, Float64}()
	for a in action_space
		min_r = typemax(Float64)
		for i in 1:100
			s = rand(b0)
			step = 0
			while (discount(pomdp)^step) > 0.01 && isterminal(pomdp, s) == false
				sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
				s = sp
				if r < min_r
					action_min_r[a] = r
					min_r = r
				end
				step += 1
			end
		end
	end

	max_min_r = typemin(Float64)
	for a in action_space
		if (action_min_r[a] > max_min_r)
			max_min_r = action_min_r[a]
		end
	end

	return max_min_r / (1 - discount(pomdp))
end