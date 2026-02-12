module TransformersPickleExt

using Transformers.HuggingFaceModels
using Pickle

function __init__()
    HuggingFaceModels.LOAD_PICKLE[] = Pickle.Torch.THload
    HuggingFaceModels.SAVE_PICKLE[] = Pickle.Torch.THsave
    HuggingFaceModels.IS_PICKLE[] = Pickle.Torch.islazy
    HuggingFaceModels.UNWRAP_PICKLE[] = unwrap_pickle_impl
end

function unwrap_pickle_impl(state)
    if Pickle.Torch.islazy(state)
        lazystate = state
        if Pickle.Torch.isloaded(lazystate)
            return lazystate.data
        else
            return lazystate()
        end
    end
    return state
end

end
