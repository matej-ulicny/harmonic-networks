from torch.utils import model_zoo

def load_pretrained(model, url):
    state_dict = model_zoo.load_url(url)
    state_dict = {k[7:]: v for k, v in state_dict.items() if 'module.' in k}
    model.load_state_dict(state_dict, strict=False)
