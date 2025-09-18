import asyncio
import logging
import re
import os
import traceback
from typing import Tuple
import joblib
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.exceptions import TelegramAPIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpamDetector:
    def __init__(self, model_path='spam_model'):
        self.model_path = model_path
        self.text_model = None
        self.feature_extractor = None
        self.is_loaded = False
        
    def load_models(self):
        try:
            self.text_model = joblib.load(f'{self.model_path}_text.pkl')
            self.feature_extractor = joblib.load(f'{self.model_path}_features.pkl')
            self.is_loaded = True
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
        text = ' '.join(text.split())
        return text
    
    def predict_spam(self, text):
        if not self.is_loaded:
            return False, 0.0
        if not text or not isinstance(text, str):
            return False, 0.0
        
        try:
            processed_text = self.preprocess_text(text)
            text_proba = self.text_model.predict_proba([processed_text])[0]
            text_confidence = text_proba[1]
            
            features = self.feature_extractor._extract_single_text_features(text)
            
            rule_score = 0.0
            if features['suspicious_pattern_count'] > 0:
                rule_score += 0.3
            if features['repeated_char_groups'] > 0:
                rule_score += 0.2
            if features['all_caps_words'] > 1:
                rule_score += 0.2
            if features['abusive_density'] > 0.1:
                rule_score += 0.3
            
            combined_score = (text_confidence * 0.7) + (min(rule_score, 1.0) * 0.3)
            is_spam = combined_score > 0.5
            
            return is_spam, combined_score
        except Exception as e:
            logger.error(f"Error in spam prediction: {e}")
            return False, 0.0

class SpamBot:
    def __init__(self, token, model_path='spam_model'):
        self.token = token
        self.spam_detector = SpamDetector(model_path)
        self.bot = Bot(token=self.token)
        self.dp = Dispatcher()
        self.setup_handlers()
        
    def setup_handlers(self):
        self.dp.message.register(self.cmd_start, Command("start"))
        self.dp.message.register(self.cmd_help, Command("help"))
        self.dp.message.register(self.check_spam)
        
    async def cmd_start(self, message: Message):
        welcome_text = (
            "🤖 I'm a spam detection bot!\n\n"
            "I automatically monitor messages in this group and remove spam.\n"
            "I detect abusive content in multiple languages.\n\n"
            "Admins are exempt from spam detection."
        )
        await message.answer(welcome_text)
        
    async def cmd_help(self, message: Message):
        help_text = (
            "ℹ️ Spam Detection Bot Help\n\n"
            "Commands:\n"
            "/start - Bot introduction\n"
            "/help - Show this help\n\n"
            "The bot detects:\n"
            "• Abusive content in any language\n"
            "• Suspicious patterns\n"
            "• Repeated characters\n"
            "• All-caps messages\n\n"
            "Messages are analyzed purely by ML models."
        )
        await message.answer(help_text)
        
    async def is_admin(self, chat_id: int, user_id: int) -> bool:
        try:
            chat_member = await self.bot.get_chat_member(chat_id, user_id)
            return chat_member.status in ['administrator', 'creator']
        except Exception:
            return False
            
    async def is_group_chat(self, message: Message) -> bool:
        return message.chat.type in ['group', 'supergroup']
        
    async def delete_message_with_retry(self, chat_id: int, message_id: int, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                await self.bot.delete_message(chat_id, message_id)
                return True
            except TelegramAPIError as e:
                if "message to delete not found" in str(e).lower():
                    return True
                elif attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                    continue
                else:
                    return False
            except Exception:
                return False
        return False
        
    async def check_spam(self, message: Message):
        if not await self.is_group_chat(message):
            return
        if message.left_chat_member or message.new_chat_members:
            return
        if not message.text:
            return
            
        try:
            if await self.is_admin(message.chat.id, message.from_user.id):
                return
        except Exception:
            pass
            
        is_spam, confidence = self.spam_detector.predict_spam(message.text)
        
        if is_spam:
            try:
                delete_success = await self.delete_message_with_retry(
                    message.chat.id, 
                    message.message_id
                )
                
                if delete_success:
                    warning_msg = await message.answer(
                        f"⚠️ Deleted spam message from {message.from_user.full_name} "
                        f"(Confidence: {confidence:.2f})"
                    )
                    
                    await asyncio.sleep(10)
                    try:
                        await warning_msg.delete()
                    except:
                        pass
                        
                    logger.info(
                        f"Deleted spam message from {message.from_user.id} "
                        f"(confidence: {confidence:.2f}): {message.text[:50]}..."
                    )
                        
            except Exception as e:
                logger.error(f"Error handling spam message: {e}")

    async def run(self):
        if not self.spam_detector.load_models():
            logger.error("Failed to load spam detection models")
            return
        logger.info("Starting spam detection bot...")
        await self.dp.start_polling(self.bot)

def main():
    BOT_TOKEN = "YOUR_BOT_TOKEN"
    
    if BOT_TOKEN == "YOUR_BOT_TOKEN":
        logger.error("Please set your bot token in the code")
        return
        
    bot = SpamBot(BOT_TOKEN)
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")

if __name__ == "__main__":
    main()
