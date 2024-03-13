
mutable struct FscNode
    _state_particles::Vector{Any}
    _Q_action::Dict{Any,Float64}
    _R_action::Dict{Any,Float64} # expected instant reward 
    _V_a_o_n::Dict{Any, Dict{Int64, Dict{Int64, Float64}}}
    _V_node_s::Dict{Any, Float64}
    _V_node_s_count::Dict{Any, Int64}
    _V_node::Float64
    _best_action_update::Dict{Any,Bool}
end

mutable struct FSC
    _eta::Vector{Dict{Pair{Any,Int64},Int64}}
    _nodes::Vector{FscNode}
    _action_space
    _obs_space
end

function InitFscNode(action_space, obs_space)
    init_particles = []
    # --- init for actions ---
    init_Q_action = Dict{Any,Float64}()
    init_R_action = Dict{Any,Float64}()
    init_V_a_o_n = Dict{Any,Dict{Int64,Dict{Int64,Float64}}}() 
    init_best_action_update = Dict{Any, Bool}()
    for a in action_space
        init_Q_action[a] = 0.0
        init_R_action[a] = 0.0
        init_V_a_o_n[a] = Dict{Int64, Dict{Int64, Float64}}()
        for o in obs_space
            init_V_a_o_n[a][o] =  Dict{Int64, Float64}()
        end
        init_best_action_update[a] = false 
    end
    init_V_node_s = Dict{Any, Float64}()
    init_V_node_s_count = Dict{Any, Int64}()
    init_V_node = 0.0
    return FscNode(init_particles,
                    init_Q_action,
                    init_R_action,
                    init_V_a_o_n,
                    init_V_node_s,
                    init_V_node_s_count,
                    init_V_node,
                    init_best_action_update)
end

function CreatNode(b, action_space, obs_space)
    node = InitFscNode(action_space, obs_space)
    node._state_particles = b
    return node
end


function InitFSC(max_node_size::Int64, action_space, obs_space)
    init_eta = Vector{Dict{Pair{Any,Int64},Int64}}(undef, max_node_size)
    for i in range(1, stop=max_node_size)
        init_eta[i] = Dict{Pair{Any,Int64},Int64}()
    end
    init_nodes = Vector{FscNode}()
    return FSC(init_eta,
                init_nodes,
                action_space,
                obs_space)
end

function GetBestAction(n::FscNode)
    Q_max = typemin(Float64)
    best_a = rand(keys(n._Q_action))
    for (key, value) in n._Q_action
        if value > Q_max
            Q_max = value
            best_a = key
        end
    end

    return best_a
end

