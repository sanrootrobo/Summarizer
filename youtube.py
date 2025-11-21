# youtube.py

import os
import pickle
import re
from datetime import datetime

# New imports for the fallback mechanism
import yt_dlp
from googleapiclient.errors import HttpError

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import config

class YouTubeService:
    """Handles all interactions with the YouTube Data API, with a yt-dlp fallback."""
    
    def __init__(self):
        self.service = self._authenticate()

    def _authenticate(self):
        creds = None
        if os.path.exists(config.TOKEN_FILE):
            with open(config.TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing token: {e}")
                    creds = None
            if not creds:
                if not os.path.exists(config.CREDENTIALS_FILE):
                    raise FileNotFoundError(f"OAuth credentials '{config.CREDENTIALS_FILE}' not found.")
                flow = InstalledAppFlow.from_client_secrets_file(config.CREDENTIALS_FILE, config.SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(config.TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
        
        return build('youtube', 'v3', credentials=creds)

    def search_videos(self, query, max_results=5, order='relevance', channel_name=None):
        """
        Performs a search on YouTube.
        Tries the official API first, and falls back to yt-dlp if the quota is exceeded.
        """
        try:
            print("Attempting search with YouTube Data API...")
            if not self.service:
                raise ConnectionError("YouTube service is not initialized.")
            
            channel_id = None
            if channel_name:
                channel_id = self._find_channel_id(channel_name)
                if not channel_id:
                    query = f"{channel_name} {query}"

            search_params = {
                'q': query,
                'part': 'id,snippet',
                'maxResults': min(max_results, config.MAX_SEARCH_RESULTS),
                'type': 'video',
                'order': order,
                'channelId': channel_id
            }
            
            search_response = self.service.search().list(**search_params).execute()
            return self._process_api_search_results(search_response)

        except HttpError as e:
            if 'quotaExceeded' in str(e):
                print("YouTube API quota exceeded. Falling back to yt-dlp...")
                return self._search_with_yt_dlp(query, max_results)
            else:
                raise e

    def _find_channel_id(self, channel_name):
        search_response = self.service.search().list(
            q=channel_name, part='id', maxResults=1, type='channel'
        ).execute()
        
        if items := search_response.get('items'):
            return items[0]['id']['channelId']
        return None

    def _process_api_search_results(self, response):
        video_ids = [item['id']['videoId'] for item in response.get('items', [])]
        if not video_ids: return []

        video_details_response = self.service.videos().list(
            part='snippet,statistics,contentDetails', id=','.join(video_ids)
        ).execute()

        videos = []
        for item in video_details_response.get('items', []):
            videos.append({
                'id': item['id'],
                'url': f"https://www.youtube.com/watch?v={item['id']}",
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'description': item['snippet'].get('description', ''),
                'duration': self._parse_iso_duration(item['contentDetails'].get('duration', 'PT0S')),
                'views': self._format_view_count(int(item['statistics'].get('viewCount', 0))),
                'published': self._format_iso_date(item['snippet'].get('publishedAt', '')),
            })
        return videos
    
    def _search_with_yt_dlp(self, query, max_results):
        """Performs a search using yt-dlp by scraping search results."""
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'force_generic_extractor': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                search_query = f"ytsearch{max_results}:{query}"
                result = ydl.extract_info(search_query, download=False)
                return self._process_yt_dlp_results(result.get('entries', []))
            except Exception as e:
                print(f"An error occurred during yt-dlp fallback search: {e}")
                return []

    def _process_yt_dlp_results(self, entries):
        """Formats yt-dlp search results to match the application's expected format."""
        videos = []
        for item in entries:
            if not item: continue
            videos.append({
                'id': item.get('id'),
                'url': item.get('webpage_url') or f"https://www.youtube.com/watch?v={item.get('id')}",
                'title': item.get('title', 'N/A'),
                'channel': item.get('channel', 'N/A'),
                'description': item.get('description', ''),
                'duration': self._parse_duration_from_seconds(item.get('duration')),
                'views': self._format_view_count(item.get('view_count', 0)),
                'published': self._format_date_from_string(item.get('upload_date')),
            })
        return videos

    @staticmethod
    def _parse_iso_duration(iso_duration):
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso_duration).groups()
        hours, minutes, seconds = [int(val) if val else 0 for val in match]
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @staticmethod
    def _format_iso_date(iso_date):
        try:
            return datetime.fromisoformat(iso_date.replace('Z', '+00:00')).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            return "Unknown"

    @staticmethod
    def _parse_duration_from_seconds(seconds):
        """Formats duration in seconds (from yt-dlp) to a readable string."""
        if seconds is None: return "N/A"
        
        # --- FIX: Cast seconds to int to prevent formatting errors ---
        seconds = int(seconds)
        
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    @staticmethod
    def _format_date_from_string(date_str):
        """Formats a 'YYYYMMDD' string (from yt-dlp) to 'YYYY-MM-DD'."""
        if date_str is None: return "Unknown"
        try:
            return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-d')
        except (ValueError, TypeError):
            return "Unknown"

    @staticmethod
    def _format_view_count(count):
        if count is None: return "N/A"
        if count >= 1_000_000: return f"{count/1_000_000:.1f}M"
        if count >= 1_000: return f"{count/1_000:.1f}K"
        return str(count)
