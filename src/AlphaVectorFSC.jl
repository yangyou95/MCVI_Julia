# Yang: What about each FSC node stores a belief tree node?
# FSC node should not contain a specific belief?
mutable struct FscNode
    _Q_action::Dict{Any,Float64}
    _R_action::Dict{Any,Float64} # expected instant reward 
    _V_a_o_n::Dict{Any, Dict{Any, Dict{Int64, Float64}}}
    _V_node::Float64 #a lower bound value
    _best_action::Any
end

mutable struct FSC
    _eta::Vector{Dict{Pair{Any,Any},Int64}}
    _nodes::Vector{FscNode}
    _action_space
    _obs_space
    _start_node_index::Int64
end

function InitFscNode(action_space, obs_space)
    # --- init for actions ---
    init_Q_action = Dict{Any,Float64}()
    init_R_action = Dict{Any,Float64}()
    init_V_a_o_n = Dict{Any,Dict{Any,Dict{Int64,Float64}}}() 
    # init_best_action_update = Dict{Any, Bool}()
    init_best_action = rand(action_space)
    for a in action_space
        init_Q_action[a] = 0.0
        init_R_action[a] = 0.0
        init_V_a_o_n[a] = Dict{Any, Dict{Int64, Float64}}()
        for o in obs_space
            init_V_a_o_n[a][o] =  Dict{Int64, Float64}()
        end
        # init_best_action_update[a] = false 
    end
    init_V_node = 0.0
    return FscNode(init_Q_action,
                    init_R_action,
                    init_V_a_o_n,
                    init_V_node,
                    init_best_action)
end

function CreatNode(action_space, obs_space)
    node = InitFscNode(action_space, obs_space)
    return node
end


function InitFSC(max_node_size::Int64, action_space, obs_space)
    init_eta = Vector{Dict{Pair{Any,Any},Int64}}(undef, max_node_size)
    for i in range(1, stop=max_node_size)
        init_eta[i] = Dict{Pair{Any,Any},Int64}()
    end
    init_nodes = Vector{FscNode}()
    return FSC(init_eta,
                init_nodes,
                action_space,
                obs_space,
                1)
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

    n._best_action = best_a
    return best_a
end