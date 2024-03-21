using POMDPs
include("Bounds.jl")
# A belief tree, the root node stores the initial belief



mutable struct BeliefTreeNode
    _state_particles::Vector{Any}
    _child_nodes::Dict{Pair{Any, Any}, BeliefTreeNode}
    _best_action::Any
    _upper_bound::Float64 # Or upper bounds over actions
    _lower_bound::Float64 # Or lower bounds over actions
    # _fsc_node_index::Int64 # link to a FSC node index
end



"""
Create new belief tree node with U and L initilizations
"""
function CreateBelieTreefNode(b_tree_node_parent::BeliefTreeNode, a, o, b_new::Vector{Any}, Q_learning_policy::Qlearning, pomdp)
    a, U = EvaluateUpperBound(b_new, Q_learning_policy)
    new_tree_node = BeliefTreeNode(b_new, Dict{Pair{Any, Any}, a, BeliefTreeNode}(), U, FindRLower(pomdp, b_new, actions(pomdp)))
    b_tree_node_parent._child_nodes[Pair(a, o)] = new_tree_node
end



"""
Sample Beliefs from a belief tree with heuristics
"""
function SampleBeliefs(root::BeliefTreeNode, b_list::Vector{Any}, nb_sim::Int64, pomdp, Q_learning_policy::Qlearning)
    # choose the best action
    a_best = root._best_action
    # choose an observation that maximize (U - L) for every b_a_o
    obs_space = observations(POMDP)
    largest_gap = typemin(Float64)
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

    # if node doesn't have childs with action a_best 
    if !bool_has_childs
        next_beliefs = BeliefUpdate(root._state_particles, a_best, nb_sim, pomdp) 
        for (o_temp, b_next) in next_beliefs
            CreateBelieTreefNode(root, a_best, o_temp, b_next)
        end
    end


    ## Sample beliefs along the tree
    if haskey(root._child_nodes[Pair(a, o)])
        push!(b_list, root._state_particles)
        SampleBeliefs(root._child_nodes[Pair(a, o)], b_list, nb_sim, pomdp)
    else 
        # Belief update and add the corresponding belief
        # need a parameter for nb_sim
        next_beliefs = BeliefUpdate(root._state_particles, a, nb_sim, pomdp) 
        for (o_temp, b_next) in next_beliefs
            CreateBelieTreefNode(root, a, o_temp, b_next)
        end
        push!(b_list, root._child_nodes[Pair(a, o)]._state_particles)
    end
end


# Update Beliefs with MC sampling
function BeliefUpdate(b, a, nb_sim::Int64, pomdp)
    next_beliefs = Dict{Any, Any}() # a map from observations to beliefs
    for i in 1:nb_sim
        s = rand(b)
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
        if haskey(next_beliefs, o)
            push!(next_beliefs[o], sp)
        else
            next_beliefs[o] = [sp]
        end
    end

    return next_beliefs 
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