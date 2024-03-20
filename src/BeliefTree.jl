using POMDPs
include("Bounds.jl")
# A belief tree, the root node stores the initial belief



mutable struct BeliefTreeNode
    _state_particles::Vector{Any}
    _child_nodes::Dict{Pair{Any, Any}, BeliefTreeNode}
    _upper_bound::Float64 # Or upper bounds over actions
    _lower_bound::Float64 # Or lower bounds over actions
end

function CreateBelieTreefNode(b_tree_node_parent::BeliefTreeNode, a, o, b_new::Vector{Any})
    new_tree_node = BeliefTreeNode(b_new, Dict{Pair{Any, Any}, BeliefTreeNode}(), 0, 0)
    b_tree_node_parent._child_nodes[Pair(a, o)] = new_tree_node
    # Evaluate Upper and Lower?
end


function SampleBeliefs(root::BeliefTreeNode, b_list::Vector{Any}, nb_sim::Int64, pomdp, Q_learning_policy::Qlearning)
    # choose the best action
    a, U = EvaluateUpperBound(root._state_particles, Q_learning_policy)
    # choose an observation that maximize (U - L) for every b_a_o
    obs_space = observations(POMDP)
    largest_gap = typemin(Float64)
    o_selected = rand(obs_space)
    bool_has_childs = false

    # if node has childs 
    for o in obs_space
        ao_edge = Pair(a, o)
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

    # if node doesn't have childs 
    if !bool_has_childs
        next_beliefs = BeliefUpdate(root._state_particles, a, nb_sim, pomdp) 
        for (o_temp, b_next) in next_beliefs
            CreateBelieTreefNode(root, a, o_temp, b_next)
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