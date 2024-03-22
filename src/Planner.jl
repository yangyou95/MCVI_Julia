include("./BeliefTree.jl")
include("./Bounds.jl")
include("./AlphaVectorFSC.jl")

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



# function BackUp(b, fsc::FSC, L::Int64, nb_sample::Int64, pomdp)
function BackUp(nI_new::Int64, fsc::FSC, RL::Float64, L::Int64, nb_sample::Int64, pomdp, action_space, obs_space)
    #  a new node (alpha-vector in MCVI)
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
    fsc._nodes[nI_new]._V_node = fsc._nodes[nI_new]._Q_action[best_a]
    # return nI_new
end



# # Expand beliefs
# function ExpandBeliefs(fsc::FSC, 
#                         nI::Int64, 
#                         s, 
#                         nb_sim::Int64,
#                         current_step::Int64, 
#                         L::Int64,
#                         pomdp, 
#                         action_space, 
#                         obs_space, 
#                         belief_node_list::Vector{Int64})

#     # Expand beliefs from root
#     if current_step < L && !isterminal(pomdp, s)
        
#         if !(nI in belief_node_list)
#             push!(belief_node_list, nI)
#             # println("add node $nI")
#         end 

#         a = GetBestAction(fsc._nodes[nI])
#         # if a != fsc._nodes[nI]._best_action && 
#             # fsc._nodes[nI]._best_action = a
#         if fsc._nodes[nI]._best_action_update[a] == false 
#             println("adding new beliefs from node $nI with action $a")
#             new_beliefs = BeliefUpdate(fsc._nodes[nI]._state_particles, a, nb_sim, pomdp)
#             for (o, b_new) in new_beliefs
#                 n_new = CreatNode(b_new, action_space, obs_space)
#                 push!(fsc._nodes, n_new)
#                 nI_new = length(fsc._nodes)
#                 push!(belief_node_list, nI_new)
#             end
#             fsc._nodes[nI]._best_action_update[a] = true
#         else
#             sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
#             n_nextI = fsc._eta[nI][Pair(a, o)]
#             # println("go one more step to node $n_nextI")
#             ExpandBeliefs(fsc, n_nextI, sp, nb_sim, current_step + 1, L, pomdp, action_space, obs_space, belief_node_list)
#         end
#     end
# end


# function ExpandBeliefsWithBestActions(fsc::FSC) 
#     nI = 1
#     open_list = [nI]
#     result_list = [nI]
#     while length(open_list) > 0
#         nI = last(open_list)
#         deleteat!(open_list, length(open_list))
#         a = GetBestAction(fsc._nodes[nI])
#         for (k,v) in fsc._eta[nI]
#             if (k[1] == a) && !(v in result_list)
#                 push!(open_list, v)
#                 push!(result_list, v)  
#             end
#         end
#     end

#     return result_list
# end


function MCVIPlanning(b0, 
                        fsc::FSC, 
                        pomdp, 
                        RL::Float64,
                        L::Int64, 
                        nb_sample::Int64, 
                        epsilon::Float64, 
                        nb_iter::Int64)

    action_space = actions(pomdp)
    obs_space = observations(pomdp)
    node = CreatNode(b0, action_space, obs_space)
    push!(fsc._nodes, node)
    nI_start = length(fsc._nodes)
    belief_node_list = [nI_start]
    # backup_list = [nI_start]
    # The solver will stop when bounds (Upper - Lower) < epsilon
    # Expand Belief first?
    # Each belief will create a new alpha vector (a new FSC node)
    # Currently just do fixed iteration

    for i in 1:nb_iter
        # Add new alpha vector node for each b in belief set
        V_root = typemin(Float64)
        # while length(belief_node_list) != 0

        println("!! Belief Expand Process !!")
        belief_node_list = Vector{Int64}()
        # Expand belief set
        ExpandBeliefs(fsc, nI_start, rand(b0), nb_sample, 0, L, pomdp, action_space, obs_space, belief_node_list)
        reverse!(belief_node_list)

        println("BackUp Process")
        while abs(V_root - fsc._nodes[nI_start]._V_node) > epsilon
            V_root = fsc._nodes[nI_start]._V_node
            # nI = pop!(belief_node_list)
            for nI in 1:length(fsc._nodes)
                # nI_input = length(fsc._nodes) + 1 - nI
                BackUp(nI, fsc, RL, L, nb_sample, pomdp, action_space, obs_space)
            end

        end

        # backup_list = ExpandBeliefsWithBestActions(fsc)
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
	println("avg sum:", EvaluateLowerBound(b0, pomdp, fsc, discount, nb_sim))
end


function EvaluateNodeBounds(node::BeliefTreeNode, pomdp, fsc::FSC, discount::Float64, nb_sim::Int64, V_mdp::Qlearning)
    a, U = EvaluateUpperBound(node._state_particles, V_mdp)
    L = EvaluateLowerBound(node._state_particles, pomdp, fsc, discount, nb_sim)
    node._upper_bound = U
    node._lower_bound = L
end