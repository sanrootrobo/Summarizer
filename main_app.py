# main_app.py

import wx
import threading
import subprocess
import webbrowser
import sys
import os
import shutil
from wx.lib.newevent import NewEvent

# Import our existing backend
import config
from youtube import YouTubeService
from agent import AssistantAgent

# --- NEW: Custom Gradient Panel ---
# This widget is responsible for drawing the smooth gradient background.
class GradientPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.gradient_start = wx.Colour(config.THEME["gradient_start"])
        self.gradient_end = wx.Colour(config.THEME["gradient_end"])

    def on_paint(self, event):
        """Called when the panel needs to be redrawn."""
        dc = wx.PaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        
        if gc:
            rect = self.GetClientRect()
            # Draw a smooth linear gradient from top to bottom
            gc.DrawLinearGradient(
                x1=0, y1=0,
                x2=0, y2=rect.height,
                c1=self.gradient_start,
                c2=self.gradient_end
            )

# --- Custom Result Panel ---
class ResultPanel(wx.Panel):
    def __init__(self, parent, video_data):
        super().__init__(parent)
        self.video = video_data
        
        # Set background to transparent so the gradient shows through.
        # This requires a bit of a trick to work on all platforms.
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.default_bg_color = wx.Colour(config.THEME["gradient_start"])
        self.default_bg_color.SetAlpha(0) # Make it fully transparent initially
        self.hover_bg_color = wx.Colour(config.THEME["bg_dark"])
        
        self.current_bg_color = self.default_bg_color
        
        self.SetCursor(wx.Cursor(wx.CURSOR_HAND))
        
        # --- Layout and Widgets ---
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        
        play_icon = wx.StaticText(self, label="▶")
        play_icon.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        play_icon.SetForegroundColour(config.THEME["accent"])

        title_font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        channel_font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        title_text = wx.StaticText(self, label=self.video['title'])
        title_text.SetFont(title_font)
        title_text.SetForegroundColour(config.THEME["fg"])
        channel_text = wx.StaticText(self, label=f"by {self.video['channel']}")
        channel_text.SetFont(channel_font)
        channel_text.SetForegroundColour(config.THEME["fg_secondary"])

        text_sizer = wx.BoxSizer(wx.VERTICAL)
        text_sizer.Add(title_text, 0, wx.EXPAND | wx.BOTTOM, 2)
        text_sizer.Add(channel_text, 0, wx.EXPAND, 0)
        
        hbox.Add(play_icon, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 15)
        hbox.Add(text_sizer, 1, wx.ALIGN_CENTER_VERTICAL, 0)
        
        vbox.Add(hbox, 1, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(vbox)
        
        # --- Event Handling ---
        self.Bind(wx.EVT_PAINT, self.on_paint_result)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_click)
        play_icon.Bind(wx.EVT_LEFT_DOWN, self.on_click)
        title_text.Bind(wx.EVT_LEFT_DOWN, self.on_click)
        channel_text.Bind(wx.EVT_LEFT_DOWN, self.on_click)
        self.Bind(wx.EVT_ENTER_WINDOW, self.on_hover)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_leave)

    def on_paint_result(self, event):
        """Draw a custom background for the result item."""
        dc = wx.PaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        if gc:
            gc.SetBrush(wx.Brush(self.current_bg_color))
            gc.SetPen(wx.TRANSPARENT_PEN)
            gc.DrawRectangle(0,0, *self.GetSize())

    def on_click(self, event):
        wx.PostEvent(self.GetParent().GetParent(), PlayVideoEvent(video=self.video))

    def on_hover(self, event):
        self.current_bg_color = self.hover_bg_color
        self.Refresh()

    def on_leave(self, event):
        self.current_bg_color = self.default_bg_color
        self.Refresh()


PlayVideoEvent, EVT_PLAY_VIDEO = NewEvent()

# --- Main Spotlight Frame ---
class SpotlightFrame(wx.Frame):
    def __init__(self):
        style = wx.FRAME_SHAPED | wx.STAY_ON_TOP
        super().__init__(None, title=config.APP_TITLE, style=style)
        
        # --- Visual Enhancements ---
        self.SetTransparent(240) # 255 is fully opaque, 0 is invisible
        self.SetBackgroundColour(wx.Colour(config.THEME["gradient_end"])) # For frame edges
        self.SetSize((800, 450))
        self.CenterOnScreen(wx.BOTH)

        self._initialize_backend()
        self._create_widgets()
        self._bind_events()

    def _initialize_backend(self):
        self.youtube_service = None
        self.agent = None
        threading.Thread(target=self._backend_init_thread, daemon=True).start()

    def _backend_init_thread(self):
        try:
            self.youtube_service = YouTubeService()
            self.agent = AssistantAgent(self.youtube_service)
            wx.CallAfter(self.status_label.SetLabel, "Ask the AI to find YouTube videos...")
        except Exception as e:
            wx.CallAfter(self.status_label.SetLabel, f"Initialization Error: {e}")

    def _create_widgets(self):
        # Use the GradientPanel as the main background
        self.panel = GradientPanel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Search Text Control - make its background match the top of the gradient
        self.search_ctrl = wx.TextCtrl(self.panel, style=wx.TE_PROCESS_ENTER | wx.BORDER_NONE)
        self.search_ctrl.SetBackgroundColour(config.THEME["gradient_start"])
        self.search_ctrl.SetForegroundColour(config.THEME["fg"])
        self.search_ctrl.SetFont(wx.Font(22, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.search_ctrl.SetHint(" Search YouTube with AI...")

        # Results Scrolled Window - also transparent to show the gradient
        self.results_win = wx.ScrolledWindow(self.panel)
        self.results_win.SetBackgroundColour(wx.Colour(0,0,0,0)) # Transparent
        self.results_sizer = wx.BoxSizer(wx.VERTICAL)
        self.results_win.SetSizer(self.results_sizer)
        self.results_win.SetScrollRate(0, 5)

        self.status_label = wx.StaticText(self.results_win, label="Initializing backend services...")
        self.status_label.SetForegroundColour(config.THEME["fg_secondary"])
        self.status_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.results_sizer.Add(self.status_label, 0, wx.ALL | wx.CENTER, 20)

        main_sizer.Add(self.search_ctrl, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(wx.StaticLine(self.panel), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        main_sizer.Add(self.results_win, 1, wx.EXPAND | wx.ALL, 10)
        self.panel.SetSizer(main_sizer)
        
    def _bind_events(self):
        self.Bind(wx.EVT_ACTIVATE, self.on_activate)
        self.search_ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_search)
        self.search_ctrl.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(EVT_PLAY_VIDEO, self.on_play_video)

    def on_activate(self, event):
        if not event.GetActive():
            self.Close()
        event.Skip()
        
    def on_key_down(self, event):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.Close()
        event.Skip()

    def on_search(self, event):
        prompt = self.search_ctrl.GetValue().strip()
        if not prompt or not self.agent: return
        
        self.results_sizer.Clear(delete_windows=True)
        status = wx.StaticText(self.results_win, label=f"🤔 Searching for: {prompt}")
        status.SetForegroundColour(config.THEME["fg_secondary"])
        self.results_sizer.Add(status, 0, wx.ALL | wx.CENTER, 20)
        self.results_win.Layout()
        
        threading.Thread(target=self._search_thread, args=(prompt,), daemon=True).start()

    def _search_thread(self, prompt):
        try:
            response = self.agent.invoke(prompt)
            videos = self.agent.last_search_results
            wx.CallAfter(self.display_results, videos)
        except Exception as e:
            wx.CallAfter(self.display_error, str(e))

    def display_results(self, videos):
        self.results_sizer.Clear(delete_windows=True)
        if not videos:
            self.display_error("No videos found matching your query.")
            return

        for video in videos:
            result_panel = ResultPanel(self.results_win, video)
            self.results_sizer.Add(result_panel, 0, wx.EXPAND | wx.BOTTOM, 5)
        
        self.results_win.Layout()
        self.results_win.FitInside()

    def display_error(self, message):
        self.results_sizer.Clear(delete_windows=True)
        error_label = wx.StaticText(self.results_win, label=f"❌ {message}")
        error_label.SetForegroundColour("#ff6b6b")
        self.results_sizer.Add(error_label, 0, wx.ALL | wx.CENTER, 20)
        self.results_win.Layout()

    def _find_mpv_path(self):
        mpv_path = shutil.which("mpv")
        if mpv_path: return mpv_path
        if sys.platform == "darwin":
            mac_app_path = "/Applications/mpv.app/Contents/MacOS/mpv"
            if os.path.exists(mac_app_path): return mac_app_path
        return None

    def on_play_video(self, event):
        url = event.video['url']
        mpv_executable = self._find_mpv_path()
        if mpv_executable:
            try:
                subprocess.Popen([mpv_executable, url])
            except Exception as e:
                webbrowser.open(url)
        else:
            webbrowser.open(url)
        self.Close()

# --- Main Application Runner ---
if __name__ == "__main__":
    app = wx.App(False)
    frame = SpotlightFrame()
    frame.Show()
    app.MainLoop()
