from pydantic import BaseSettings

class ModelSettings(BaseSettings):
    # logging settings
    wandb_active: bool = True
    wandb_project_name: str = "codon_transformer"
    checkpoint_interval: int = 500
    checkpoint_dir: str = "codon_transformer"

    # data settings
    tokenizer_file: str = "codon_wordlevel_100vocab.json"
    train_file: str = "mdh_codon_spaces_full_train.txt"
    val_file: str = "mdh_codon_spaces_full_val.txt"
    test_file: str = "mdh_codon_spaces_full_test.txt"

    # model settings
    use_pretrained: bool = True
    batch_size: int = 4
    epochs: int = 5

if __name__ == "__main__":
    settings = ModelSettings()
    settings.yaml("settings_template.yaml")