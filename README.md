# dqn_chrome_dino
Playing Chrome Dino Game with DQN, using python 3.10.18

## Training

Run all the cells in `dqn_dino.ipynb`

## Playing
### Human mode
```bash
python dino.py human
```
### AI mode
```bash
python dino.py ai -m <model_path>

# Example

python dino.py ai -m ./models/dino_final_model.pth
```

## References

The codes are based on repo [aome510/chrome-dino-game-rl](https://github.com/aome510/chrome-dino-game-rl)
