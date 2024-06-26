include("./BeliefTree.jl")
include("./Bounds.jl")
include("./AlphaVectorFSC.jl")


# use heuristics here? Vmdp policy?
# should we have leaf nodes here? trajectory should be in circles?
function SimulateTrajectory(nI::Int64, fsc::FSC, s::Any, L::Int64, RL::Float64, pomdp) 
    gamma = discount(pomdp)
    V_n_s = 0.0
    nI_current = nI
    s_sim = s
    for step in 0:L
        if (isterminal(pomdp, s_sim))
			break
		end
        # a = rand(actions(pomdp))
        if nI_current != -1
            a = GetBestAction(fsc._nodes[nI_current])
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s_sim, a)
            if haskey(fsc._eta[nI_current], Pair(a, o))
                nI_current = fsc._eta[nI_current][Pair(a, o)]
            else
                nI_current = -1 
            end 
        else
            r = (gamma^L)*RL
            V_n_s += (gamma^step)*r
            break
            # sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
        end
        V_n_s += (gamma^step)*r
        s_sim = sp
    end
    return V_n_s
end

# function FindMaxValueNode(n::FscNode, fsc_node_list::Vector{Int64}, fsc::FSC, a, o)
function FindMaxValueNode(n::FscNode, fsc::FSC, a, o)
    max_V = typemin(Float64)
    max_nI = 1
    for nI in 1:length(fsc._nodes)
    # for nI in fsc_node_list
        V_temp = n._V_a_o_n[a][o][nI]
        if V_temp > max_V
            max_V = V_temp
            max_nI = nI
        end
    end
    return max_V, max_nI
end

# function BackUp(Tr_node::BeliefTreeNode, fsc_node_list::Vector{Int64}, fsc::FSC, RL::Float64, L::Int64, nb_sample::Int64, pomdp, action_space, obs_space)
function BackUp(Tr_node::BeliefTreeNode, fsc::FSC, RL::Float64, L::Int64, nb_sample::Int64, pomdp, action_space, obs_space)
    belief = Tr_node._state_particles
    n_new_temp = CreatNode(action_space, obs_space)
    gamma = discount(pomdp)
    for a in action_space
        for o in obs_space
            for nI in 1:length(fsc._nodes)
            # for nI in fsc_node_list
                n_new_temp._V_a_o_n[a][o][nI] = 0.0
            end
        end
    end

    temp_eta = Dict{Pair{Any, Any}, Int64}()
    for a in action_space
        # Multi-threading process can be added here
        for i in 1:nb_sample
            s = rand(belief)
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
            n_new_temp._R_action[a] += r
            for nI in 1:length(fsc._nodes)
            # for nI in fsc_node_list
                V_nI_sp = SimulateTrajectory(nI, fsc, sp, L, RL, pomdp) 
                n_new_temp._V_a_o_n[a][o][nI] += V_nI_sp
            end 
        end

        for o in obs_space
            # V_a_o, nI_a_o = FindMaxValueNode(n_new_temp, fsc_node_list, fsc, a, o)
            V_a_o, nI_a_o = FindMaxValueNode(n_new_temp, fsc, a, o)
            temp_eta[Pair(a, o)] = nI_a_o
            n_new_temp._Q_action[a] += gamma*V_a_o
        end
        n_new_temp._Q_action[a] += n_new_temp._R_action[a]
        n_new_temp._Q_action[a] /= nb_sample
    end

    best_a = GetBestAction(n_new_temp)
    V_lower = n_new_temp._Q_action[best_a]
    n_new_temp._V_node = V_lower
    Tr_node._best_action = best_a
    Tr_node._lower_bound = V_lower
    nI = FindOrInsertNode(n_new_temp, temp_eta, obs_space, fsc)
    Tr_node._fsc_node_index = nI
end


function MCVIPlanning(b0, 
                    fsc::FSC, 
                    pomdp, 
                    RL::Float64,
                    L::Int64, 
                    nb_sample::Int64, 
                    epsilon::Float64, 
                    max_nb_iter::Int64,
                    Q_learning_policy::Qlearning,
                    Tr_root::BeliefTreeNode)

    action_space = actions(pomdp)
    obs_space = observations(pomdp)
    node = CreatNode(action_space, obs_space)
    push!(fsc._nodes, node)
    nI_start = length(fsc._nodes)
    Tr_root._fsc_node_index = nI_start
    gamma = discount(pomdp)

    i = 0
    while i < max_nb_iter
        println("--- Iter $i ---")
        UpdateUpperBound(Tr_root, gamma, 0)
        println("Tr_root upper bound:", Tr_root._upper_bound)
        println("Tr_root lower bound:", Tr_root._lower_bound)
        precision = Tr_root._upper_bound - Tr_root._lower_bound
        println("Precision:", precision)
        if precision < epsilon
            println("MCVI Planning finished, reached to the target precision")
            break
        end
        belief_tree_node_list = []
        belief_sample_time_taken = @elapsed SampleBeliefs(Tr_root, rand(b0), 0, L, nb_sample, pomdp, Q_learning_policy, belief_tree_node_list)
        println("Belief Expand Process takes $belief_sample_time_taken seconds")

        # this is meant to prune FSC nodes, only select dominiated nodes for backup, seems not work properly
        # fsc_node_list = Vector{Int64}()
        # GetFscNodeList(Tr_root, fsc_node_list) 

        backup_elapsed_time = @elapsed begin
            while length(belief_tree_node_list) != 0
                Tr_node = pop!(belief_tree_node_list)
                BackUp(Tr_node, fsc, RL, L, nb_sample, pomdp, action_space, obs_space)
            end
        end
        println("BackUp Process takes $backup_elapsed_time seconds")
        i += 1
        if precision < epsilon
            println("MCVI Planning finished, reached to the max number of iteration")
            break
        end
    end

    fsc._start_node_index = Tr_root._fsc_node_index
    return fsc
end


function SimulationWithFSC(b0, pomdp, fsc::FSC, steps::Int64)
	s = rand(b0)
	sum_r = 0.0
	nI = fsc._start_node_index
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
	nI_true = fsc._start_node_index
    for sim_i in 1:nb_sim
        s = rand(b0)
        sum_sim_i = 0.0
        step = 0
        nI = nI_true
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

function EvaluationWithSimulationFSC(b0, pomdp, fsc::FSC, discount::Float64, nb_sim::Int64, L::Int64, a_blind)
	# println("avg sum:", EvaluateLowerBound(b0, pomdp, fsc, discount, nb_sim))
	sum_r = 0.0
    nI_true = fsc._start_node_index
    for sim_i in 1:nb_sim
        s = rand(b0)
        sum_sim_i = 0.0
        step = 0
        nI = nI_true
        while L > step
            if (isterminal(pomdp, s))
                break
            end

            if nI != -1
                a = GetBestAction(fsc._nodes[nI])
            else 
                a = a_blind
            end
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
            s = sp
            sum_sim_i += (discount^step) * r
            edge = Pair(a, o)
            if haskey(fsc._eta[nI], edge)
                nI = fsc._eta[nI][edge]
            else
                nI = -1
            end
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
function FindOrInsertNode(temp_node::FscNode, temp_eta::Dict{Pair{Any, Any}, Int64}, obs_space, fsc::FSC)
    
    for nI in 1:length(fsc._nodes)
        # First check the best action
        if fsc._nodes[nI]._best_action != temp_node._best_action
            continue
        else 
            for o in obs_space
                temp_edge = Pair(temp_node._best_action, o)
                if !haskey(fsc._eta[nI], temp_edge)
                    return InsertNode(temp_node, temp_eta, fsc)
                else
                    if fsc._eta[nI][temp_edge] != temp_eta[temp_edge]
                        return InsertNode(temp_node, temp_eta, fsc)
                    end
                end
            end
            # find same node
            return nI
        end 
    end
    return InsertNode(temp_node, temp_eta, fsc)

end

function InsertNode(temp_node::FscNode, temp_eta::Dict{Pair{Any, Any}, Int64}, fsc::FSC)
    push!(fsc._nodes, temp_node)
    nI_new = length(fsc._nodes)
    fsc._eta[nI_new] = temp_eta
    return nI_new
end