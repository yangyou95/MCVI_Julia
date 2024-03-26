include("./BeliefTree.jl")
include("./Bounds.jl")
include("./AlphaVectorFSC.jl")


# use heuristics here? Vmdp policy?
# should we have leaf nodes here? trajectory should be in circles?
function SimulateTrajectory(nI::Int64, fsc::FSC, s::Any,  L::Int64, pomdp) 
    gamma = discount(pomdp)
    V_n_s = 0.0
    nI_current = nI
    for step in 0:L
        if (isterminal(pomdp, s))
			break
		end
        a = rand(actions(pomdp))
        if nI_current != -1
            a = GetBestAction(fsc._nodes[nI_current])
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
            if haskey(fsc._eta[nI_current], Pair(a, o))
                nI_current = fsc._eta[nI_current][Pair(a, o)]
            else
                nI_current = -1 
            end 
        else 
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
        end
        V_n_s += (gamma^step)*r
        s = sp
    end
    return V_n_s
end

# function SimulateTrajectory(nI::Int64, fsc::FSC, s::Any,  L::Int64, pomdp) 
#     gamma = discount(pomdp)
#     V_n_s = 0.0
#     nI_current = nI
#     for step in 0:L
#         if (isterminal(pomdp, s))
# 			break
# 		end
#         a = GetBestAction(fsc._nodes[nI_current])
#         sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
#         nI_current = fsc._eta[nI_current][Pair(a, o)]
#         V_n_s += (gamma^step)*r
#         s = sp
#     end
#     return V_n_s
# end

function FindMaxValueNode(n:: FscNode, fsc::FSC, a, o)
    max_V = typemin(Float64)
    max_nI = 1
    for nI in 1:length(fsc._nodes)
        V_temp = n._V_a_o_n[a][o][nI]
        if V_temp > max_V
            max_V = V_temp
            max_nI = nI
        end
    end
    return max_V, max_nI
end


function BackUp(Tr_node::BeliefTreeNode, fsc::FSC, RL::Float64, L::Int64, nb_sample::Int64, pomdp, action_space, obs_space)
    # check if Tr_node is already processed before
    nI_new = Tr_node._fsc_node_index
    if nI_new == -1
        # not processed before, add a new fsc node to G
        n_new = CreatNode(Tr_node._state_particles, action_space, obs_space)
        push!(fsc._nodes, n_new)
        nI_new = length(fsc._nodes)
        Tr_node._fsc_node_index = nI_new
    end
    
    gamma = discount(pomdp)
    V_nI_new = fsc._nodes[nI_new]._V_node
    println("nI_new $nI_new, V $V_nI_new")
    for a in action_space
        fsc._nodes[nI_new]._R_action[a] = 0.0
        fsc._nodes[nI_new]._Q_action[a] = 0.0 #  Do we need init Q?
        for o in obs_space
            for nI in 1:length(fsc._nodes)
                fsc._nodes[nI_new]._V_a_o_n[a][o][nI] = 0.0
            end
        end
    end

    for a in action_space
        for i in 1:nb_sample
            s = rand(fsc._nodes[nI_new]._state_particles)
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
            fsc._nodes[nI_new]._R_action[a] += r
            for nI in 1:length(fsc._nodes)
                V_nI_sp = SimulateTrajectory(nI, fsc, sp, L, pomdp) 
                fsc._nodes[nI_new]._V_a_o_n[a][o][nI] += V_nI_sp
            end 
        end

        for o in obs_space
            V_a_o, nI_a_o = FindMaxValueNode(fsc._nodes[nI_new], fsc, a, o)
            # fsc._eta[nI_new][a][o] = nI_a_o
            fsc._eta[nI_new][Pair(a, o)] = nI_a_o
            fsc._nodes[nI_new]._Q_action[a] += discount(pomdp)*V_a_o
        end
        fsc._nodes[nI_new]._Q_action[a] += fsc._nodes[nI_new]._R_action[a]
        fsc._nodes[nI_new]._Q_action[a] /= nb_sample
    end

    best_a = GetBestAction(fsc._nodes[nI_new])
    V_lower = fsc._nodes[nI_new]._Q_action[best_a]
    fsc._nodes[nI_new]._V_node = V_lower

    Tr_node._best_action = best_a
    Tr_node._lower_bound = V_lower
end


function MCVIPlanning(b0, 
                    fsc::FSC, 
                    pomdp, 
                    RL::Float64,
                    L::Int64, 
                    nb_sample::Int64, 
                    epsilon::Float64, 
                    nb_iter::Int64,
                    Q_learning_policy::Qlearning,
                    Tr_root::BeliefTreeNode)

    action_space = actions(pomdp)
    obs_space = observations(pomdp)
    node = CreatNode(b0, action_space, obs_space)
    push!(fsc._nodes, node)
    nI_start = length(fsc._nodes)
    Tr_root._fsc_node_index = nI_start


    for i in 1:nb_iter # should be iteration until convergence of (U-L)
        # Add new alpha vector node for each b in belief set
        V_root = typemin(Float64)
        println("--- Iter $i ---")
        println("Belief Expand Process")
        belief_tree_node_list = []
        SampleBeliefs(Tr_root, rand(b0), 0, L, nb_sample, pomdp, Q_learning_policy, belief_tree_node_list)
        println("BackUp Process")
        while length(belief_tree_node_list) != 0
            Tr_node = pop!(belief_tree_node_list)
            # nI_input = length(fsc._nodes) + 1 - nI
            BackUp(Tr_node, fsc, RL, L, nb_sample, pomdp, action_space, obs_space)
        end
        # for j in 1:10
        # belief_tree_node_list_temp = copy(belief_tree_node_list)
        # while length(belief_tree_node_list_temp) != 0
        #     Tr_node = pop!(belief_tree_node_list_temp)
        #     # nI_input = length(fsc._nodes) + 1 - nI
        #     BackUp(Tr_node, fsc, RL, L, nb_sample, pomdp, action_space, obs_space)
        # end
        # end
    end
    return fsc
end


function SimulationWithFSC(b0, pomdp, fsc::FSC, steps::Int64)
	s = rand(b0)
	sum_r = 0.0
	nI = 1
	for i in 1:steps
		if (isterminal(pomdp, s))
			break
		end

		println("---------")
		println("step: ", i)
		println("state:", s)
		a = GetBestAction(fsc._nodes[nI])
		println("perform action:", a)
		sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
		s = sp
		sum_r += (discount(pomdp)^i) * r
		println("recieve obs:", o)
		println("nI:", nI)
		println("nI value:", fsc._nodes[nI]._V_node)
		nI = fsc._eta[nI][Pair(a, o)]
		println("reward:", r)
	end

	println("sum_r:", sum_r)
end


function EvaluationWithSimulationFSC(b0, pomdp, fsc::FSC, discount::Float64, nb_sim::Int64)
	# println("avg sum:", EvaluateLowerBound(b0, pomdp, fsc, discount, nb_sim))
	sum_r = 0.0
	nI = 1
    for sim_i in 1:nb_sim
        s = rand(b0)
        sum_sim_i = 0.0
        step = 0
        while (discount^step) > 0.01
            if (isterminal(pomdp, s))
                break
            end
            a = GetBestAction(fsc._nodes[nI])
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
            s = sp
            sum_sim_i += (discount^step) * r
            nI = fsc._eta[nI][Pair(a, o)]
            step += 1
        end
        sum_r += sum_sim_i
    end

	println("sum_r:", sum_r / nb_sim)

end


# function EvaluateNodeBounds(node::BeliefTreeNode, pomdp, fsc::FSC, discount::Float64, nb_sim::Int64, V_mdp::Qlearning)
#     a, U = EvaluateUpperBound(node._state_particles, V_mdp)
#     L = EvaluateLowerBound(node._state_particles, pomdp, fsc, discount, nb_sim)
#     node._upper_bound = U
#     node._lower_bound = L
# end


"""
Find a same node that has a same best action and outgoing edges
"""
function FindSameNode(temp_node::FscNode, fsc::FSC)
    # codes 
end