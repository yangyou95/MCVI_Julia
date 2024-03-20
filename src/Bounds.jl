include("./AlphaVectorFSC.jl")


using POMDPModels
using POMDPs
using POMDPModelTools
using Random


# We use a Q-learning method to compute Vmdp
mutable struct Qlearning
    _Q_table::Dict{Any, Dict{Any, Float64}} #s -> a -> Q
    _learning_rate::Float64
    _explore_rate::Float64
    _action_space
    _R_max::Float64
    _R_min::Float64
end

# Upper bound value can be provided by solving a corresponding MDP
function EvaluateUpperBound(b, Q_learning_policy::Qlearning)
    max_value = typemin(Float64)
    a_best = rand(Q_learning_policy._action_space)

    for (a, value) in Q_learning_policy._action_space
        temp_value = 0.0
        for s in b
            temp_value += pb * GetQ(Q_learning_policy, s, a)
        end

        temp_value /= length(b) 
        if temp_value > max_value
            max_value = temp_value
            a_best = a 
        end
    end

    return a_best, max_value
end 


function EvaluateLowerBound(b, pomdp, fsc::FSC, discount::Float64, nb_sim::Int64)
	sum_r = 0.0
	for sim_i in 1:nb_sim
		step = 0
		sum_r_sim_i = 0.0
		s = rand(b)
		nI = 1
		bool_random_pi = false

		while (discount^step) > 0.01 && isterminal(pomdp, s) == false


			if nI == -1
				bool_random_pi = true
			end

			if bool_random_pi
				a = rand(fsc._action_space)
			else
				a = GetBestAction(fsc._nodes[nI])
			end

			sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
			s = sp
			sum_r_sim_i += (discount^step) * r

			if haskey(fsc._eta[nI], Pair(a, o))
				nI = fsc._eta[nI][Pair(a, o)]
			else
				bool_random_pi = true
			end
			step += 1
		end
		sum_r += sum_r_sim_i
	end

	return sum_r / nb_sim
end





function ChooseActionQlearning(Q_learning_Policy::Qlearning, s)
    a_selected = -1
    rand_num = rand()
    if rand_num < Q_learning_Policy._explore_rate
        a_selected = rand(Q_learning_Policy._action_space)
    else
        a_selected = BestAction(Q_learning_Policy::Qlearning, s)
    end
    return a_selected
end

function MaxQ(Q_learning_Policy::Qlearning, s)
    max_Q = typemin(Float64)
    for a in Q_learning_Policy._action_space
        Q_temp = GetQ(Q_learning_Policy, s, a)
        if Q_temp > max_Q
            max_Q = Q_temp
        end
    end
    return max_Q
end

function BestAction(Q_learning_Policy::Qlearning, s)
    max_Q = typemin(Float64)
    a_max_Q = -1
    for a in Q_learning_Policy._action_space
        Q_temp = GetQ(Q_learning_Policy, s, a)
        if Q_temp > max_Q
            a_max_Q = a
            max_Q = Q_temp
        end
    end
    return a_max_Q
end

function GetQ(Q_learning_Policy::Qlearning, s, a)
    if haskey(Q_learning_Policy._Q_table, s)
        return Q_learning_Policy._Q_table[s][a]
    else
        Q_learning_Policy._Q_table[s] = Dict{Any, Float64}()
        for a in Q_learning_Policy._action_space
            Q_learning_Policy._Q_table[s][a] = 0.0
        end
        return 0.0
    end
end

function UpdateRmaxRmin(Q_learning_Policy::Qlearning, r::Float64)
    if r > Q_learning_Policy._R_max
        Q_learning_Policy._R_max = r
    end

    if r < Q_learning_Policy._R_min
        Q_learning_Policy._R_min = r
    end
end
function EstiValueQlearning(Q_learning_Policy::Qlearning, nb_sim::Int64, s_input, pomdp)
    a_selected = -1
    gamma = discount(pomdp)
    for i in nb_sim
        step = 0
        s = deepcopy(s_input)
        while (gamma^step) > 0.01 && isterminal(pomdp, s) == false
            a_selected = ChooseActionQlearning(Q_learning_Policy, s)
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a_selected)
            UpdateRmaxRmin(Q_learning_Policy, r)
            old_Q = GetQ(Q_learning_Policy, s, a_selected) 
            new_Q = old_Q + Q_learning_Policy._learning_rate * (r + gamma * MaxQ(Q_learning_Policy, sp) - old_Q)
            Q_learning_Policy._Q_table[s][a_selected] = new_Q
            s = sp
            step += 1
        end
    end

    return MaxQ(Q_learning_Policy, s_input)
end

function Training(Q_learning_Policy::Qlearning, nb_episode_size::Int64, nb_max_episode::Int64, nb_sim::Int64, epsilon::Float64, b0, pomdp)
    improvement = typemax(Float64)
    current_avg_value = typemin(Float64)
    i_episode = 0
    while (improvement > epsilon) && (i_episode < nb_max_episode)
        println("------ Episode: ", i_episode , " ------")
        value_episode = 0.0
        for i in 1:nb_episode_size
            sum_value_tmp = 0.0
            for s in b0 
                sum_value_tmp += EstiValueQlearning(Q_learning_Policy, nb_sim, s, pomdp)
            end
            value_episode += sum_value_tmp/length(b0)
        end
        value_episode /= nb_episode_size
        improvement = value_episode - current_avg_value
        current_avg_value = value_episode
        println("Avg Value: ", current_avg_value)
        i_episode += 1
    end
end
