#!/usr/bin/env python3
"""
Test script for YouTube playlist downloading with yt-dlp
This script helps debug playlist downloading issues in MagicSample
"""

import sys
import os
import tempfile
import yt_dlp
import re

def test_youtube_playlist(url):
    """Test YouTube playlist downloading"""
    print(f"Testing YouTube playlist: {url}")
    print("=" * 50)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="test_youtube_")
    print(f"Using temporary directory: {temp_dir}")
    
    # Configure yt-dlp options (same as in MagicSample)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
        'ignoreerrors': True,
        'nocheckcertificate': True,
        'prefer_ffmpeg': True,
        'geo_bypass': True,
        'extractor_retries': 5,
        'fragment_retries': 5,
        'retries': 5,
        'verbose': True,
        # Playlist-specific options
        'playlist_items': '1-',
        'playlist_reverse': False,
        'playlist_random': False,
        'playlist_start': 1,
        'playlist_end': None,
        # Additional options for better compatibility
        'no_check_certificate': True,
        'http_chunk_size': 10485760,
        'buffersize': 1024,
        'sleep_interval': 1,
        'max_sleep_interval': 5,
        'sleep_interval_requests': 1,
        'max_sleep_interval_requests': 5,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("1. Testing info extraction...")
            
            # First, try to extract info without downloading
            try:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    print("❌ Failed to extract info - returned None")
                    return False
                
                print(f"✅ Successfully extracted info")
                print(f"   Title: {info.get('title', 'Unknown')}")
                print(f"   Type: {'Playlist' if 'entries' in info else 'Single Video'}")
                
                if 'entries' in info:
                    print(f"   Entries: {len(info['entries'])} videos")
                    print(f"   Playlist ID: {info.get('id', 'Unknown')}")
                    
                    # Show first few entries
                    for i, entry in enumerate(info['entries'][:3]):
                        if entry:
                            print(f"   Entry {i+1}: {entry.get('title', 'Unknown')}")
                            print(f"     URL: {entry.get('webpage_url', 'Unknown')}")
                
            except Exception as e:
                print(f"❌ Info extraction failed: {e}")
                print(f"   Error type: {type(e).__name__}")
                return False
            
            print("\n2. Testing actual download...")
            
            # Try to download
            try:
                ydl.download([url])
                print("✅ Download completed")
                
                # Check what files were downloaded
                downloaded_files = []
                for file in os.listdir(temp_dir):
                    if file.endswith('.wav'):
                        file_path = os.path.join(temp_dir, file)
                        file_size = os.path.getsize(file_path)
                        downloaded_files.append((file, file_size))
                        print(f"   Downloaded: {file} ({file_size} bytes)")
                
                if downloaded_files:
                    print(f"✅ Successfully downloaded {len(downloaded_files)} files")
                    return True
                else:
                    print("❌ No files were downloaded")
                    return False
                    
            except Exception as e:
                print(f"❌ Download failed: {e}")
                print(f"   Error type: {type(e).__name__}")
                return False
                
    except Exception as e:
        print(f"❌ General error: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False
    
    finally:
        # Clean up
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary directory")
        except Exception as e:
            print(f"⚠️  Failed to clean up: {e}")

def test_alternative_method(url):
    """Test alternative playlist download method"""
    print(f"\nTesting alternative method for: {url}")
    print("=" * 50)
    
    temp_dir = tempfile.mkdtemp(prefix="test_alt_")
    
    # Use flat extraction first
    alt_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(alt_opts) as ydl:
            print("1. Testing flat extraction...")
            
            flat_info = ydl.extract_info(url, download=False)
            if not flat_info or 'entries' not in flat_info:
                print("❌ Flat extraction failed")
                return False
            
            print(f"✅ Flat extraction successful")
            print(f"   Found {len(flat_info['entries'])} entries")
            
            # Show first few entries
            for i, entry in enumerate(flat_info['entries'][:3]):
                if entry:
                    print(f"   Entry {i+1}: {entry.get('title', 'Unknown')}")
                    print(f"     URL: {entry.get('url', 'Unknown')}")
            
            print("\n2. Testing individual downloads...")
            
            # Try downloading first 2 videos individually
            download_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': False,
                'ignoreerrors': True,
            }
            
            successful_downloads = 0
            for i, entry in enumerate(flat_info['entries'][:2]):  # Test first 2 only
                if entry is None:
                    continue
                
                video_url = entry.get('url') or entry.get('webpage_url')
                if not video_url:
                    continue
                
                print(f"   Downloading video {i+1}: {entry.get('title', 'Unknown')}")
                
                try:
                    with yt_dlp.YoutubeDL(download_opts) as video_ydl:
                        video_ydl.download([video_url])
                    
                    # Check if file was downloaded
                    for file in os.listdir(temp_dir):
                        if file.endswith('.wav'):
                            file_path = os.path.join(temp_dir, file)
                            file_size = os.path.getsize(file_path)
                            print(f"     ✅ Downloaded: {file} ({file_size} bytes)")
                            successful_downloads += 1
                            break
                    else:
                        print(f"     ❌ No file downloaded for video {i+1}")
                        
                except Exception as e:
                    print(f"     ❌ Failed to download video {i+1}: {e}")
            
            print(f"\n✅ Alternative method: {successful_downloads}/2 videos downloaded successfully")
            return successful_downloads > 0
            
    except Exception as e:
        print(f"❌ Alternative method failed: {e}")
        return False
    
    finally:
        # Clean up
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory")
        except Exception as e:
            print(f"⚠️  Failed to clean up: {e}")

def main():
    """Main test function"""
    print("YouTube Playlist Download Test")
    print("=" * 50)
    
    # Test URLs
    test_urls = [
        "https://www.youtube.com/playlist?list=PLrAXtmRdnEQy6nuLMHjMZOz59Oq8W0V8i",  # Example playlist
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Single video
    ]
    
    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"Testing URL: {url}")
        print(f"{'='*60}")
        
        # Test main method
        success = test_youtube_playlist(url)
        
        if not success and 'playlist' in url.lower():
            # Try alternative method for playlists
            test_alternative_method(url)
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print("If you see errors, check:")
    print("1. yt-dlp version (run: pip show yt-dlp)")
    print("2. FFmpeg installation (run: ffmpeg -version)")
    print("3. Internet connection")
    print("4. YouTube URL format")

if __name__ == "__main__":
    main()
