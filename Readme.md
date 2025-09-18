# Telegram Spam Detection Bot

An AI-powered Telegram bot that automatically detects and removes spam messages in group chats. Built with machine learning to identify abusive content, suspicious patterns, and spam messages in multiple languages including Hindi roman script.

## Features

- 🤖 **AI-Powered Detection**: Uses machine learning to identify spam content
- 🌍 **Multilingual Support**: Works with English, Hindi roman script, and other languages
- 🔍 **Pattern Recognition**: Detects suspicious character patterns and repeated characters
- ⚡ **Real-time Protection**: Automatically monitors and removes spam messages
- 👮 **Admin Protection**: Exempts administrators from spam detection
- 📊 **Confidence Scoring**: Shows confidence level for each spam detection

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Harshit-shrivastav/spam-detector-bot.git
cd spam-detector-bot
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Get Telegram Bot Token**:
   - Talk to [@BotFather](https://t.me/BotFather) on Telegram
   - Create a new bot and copy the API token
   - Replace `YOUR_BOT_TOKEN` in `main.py` with your token

## Usage

1. **Prepare your dataset**:
   Create `spam_dataset.csv` with your training data (see sample format below)

2. **Train the model**:
```bash
python train_spam_model.py
```

3. **Run the bot**:
```bash
python main.py
```

4. **Add bot to your Telegram group**:
   - Add the bot as administrator
   - Enable "Delete messages" permission

## Dataset Format

Create `spam_dataset.csv` with the following format:

```csv
text,label
"Hello, how are you today?",0
"Great meeting you yesterday",0
"This is a wonderful opportunity",0
"Free money, no strings attached",1
"Urgent! Claim your prize now",1
"Looking forward to our collaboration",0
"Limited time offer, act now",1
```

- **Label 0**: Legitimate message (ham)
- **Label 1**: Spam/abusive message

## How It Works

The bot uses a combination of:

1. **TF-IDF Vectorization**: Converts text to numerical features
2. **Logistic Regression**: Machine learning model for classification
3. **Feature Engineering**: Additional features like:
   - Suspicious character patterns
   - Repeated characters
   - All-caps words
   - URL detection
   - Mention/hashtag counting

## Configuration

Adjust the spam detection sensitivity in `bot.py`:
```python
is_spam = combined_score > 0.5  # Lower = more sensitive
```

## Sample Dataset

```csv
text,label
"kya haal chal",0
"mai badhiya",0
"ye kya bakwaas hai",1
"bahut achha",0
"tumhari maa ki",1
"meeting kab hai",0
"project deadline kya hai",0
"free paisa milta hai",1
"kal milte hain",0
"nikal yha se",1
"thanks for help",0
"visit this website",1
"how are you",0
"urgent money needed",1
"good morning",0
```

## Requirements

- Python 3.8+
- Telegram Bot Token
- Admin rights in target Telegram group

## Dependencies

- aiogram >= 3.0.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- joblib >= 1.1.0

## Author

[Harshit Shrivastav](https://github.com/Harshit-shrivastav)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

For support, please open an issue on the GitHub repository.
