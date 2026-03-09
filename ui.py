#!/usr/bin/env python3
"""
Touchscreen UI module for Raspberry Pi Vision AI
Tkinter-based interface with touch controls
"""

import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class TouchUI:
    """Touchscreen UI controller"""
    
    def __init__(self, display_config, detection_config, 
                 on_model_change=None, on_settings_change=None, on_capture=None):
        """Initialize the touchscreen UI"""
        self.display_config = display_config
        self.detection_config = detection_config
        self.on_model_change = on_model_change
        self.on_settings_change = on_settings_change
        self.on_capture = on_capture
        
        # UI state
        self.current_frame = None
        self.detections = []
        self.fps = 0
        self.show_settings = False
        self.settings_window = None
        
        # Touch tracking
        self.touch_start = None
        self.touch_threshold = 50  # pixels for swipe detection
        
        # Colors
        self.colors = {
            'bg': '#1a1a2e',
            'fg': '#ffffff',
            'accent': '#00d4ff',
            'success': '#00ff88',
            'warning': '#ffaa00',
            'danger': '#ff4444'
        }
        
        # Initialize Tkinter
        self.root = tk.Tk()
        self.root.title("Raspberry Pi Vision AI")
        
        # Set fullscreen if configured
        if display_config.get('fullscreen', False):
            self.root.attributes('-fullscreen', True)
            
        # Set window size
        width = display_config.get('width', 800)
        height = display_config.get('height', 480)
        self.root.geometry(f"{width}x{height}")
        
        # Configure styling
        self._setup_styles()
        
        # Build UI
        self._build_ui()
        
        # Bind touch/mouse events
        self._bind_events()
        
        # Start update loop
        self.running = True
        
    def _setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', 
                       background=self.colors['bg'],
                       foreground=self.colors['fg'],
                       font=('Helvetica', 16, 'bold'))
        
        style.configure('Info.TLabel',
                       background=self.colors['bg'],
                       foreground=self.colors['accent'],
                       font=('Helvetica', 10))
        
    def _build_ui(self):
        """Build the main UI"""
        # Main container
        self.main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video display canvas
        self.canvas = tk.Canvas(
            self.main_frame,
            bg=self.colors['bg'],
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Overlay frame (for detection info)
        self.overlay_frame = tk.Frame(self.main_frame, bg=self.colors['bg'])
        self.overlay_frame.place(relx=0.02, rely=0.02, anchor='nw')
        
        # FPS label
        self.fps_label = tk.Label(
            self.overlay_frame,
            text="FPS: 0",
            bg=self.colors['bg'],
            fg=self.colors['success'],
            font=('Helvetica', 12, 'bold')
        )
        self.fps_label.pack(anchor='nw')
        
        # Model label
        self.model_label = tk.Label(
            self.overlay_frame,
            text="Model: Loading...",
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            font=('Helvetica', 10)
        )
        self.model_label.pack(anchor='nw', pady=(5, 0))
        
        # Detection count label
        self.detection_label = tk.Label(
            self.overlay_frame,
            text="Objects: 0",
            bg=self.colors['bg'],
            fg=self.colors['accent'],
            font=('Helvetica', 10)
        )
        self.detection_label.pack(anchor='nw', pady=(5, 0))
        
        # Status bar at bottom
        self.status_frame = tk.Frame(self.main_frame, bg=self.colors['bg'])
        self.status_frame.place(relx=0.5, rely=0.98, anchor='s')
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Tap: Toggle | Swipe L/R: Change Model | Swipe Up: Settings",
            bg=self.colors['bg'],
            fg='#666666',
            font=('Helvetica', 8)
        )
        self.status_label.pack()
        
    def _bind_events(self):
        """Bind touch and mouse events"""
        # Mouse events (works with touchscreen)
        self.canvas.bind('<Button-1>', self._on_touch_start)
        self.canvas.bind('<B1-Motion>', self._on_touch_move)
        self.canvas.bind('<ButtonRelease-1>', self._on_touch_end)
        
        # Keyboard events
        self.root.bind('<Escape>', lambda e: self._quit())
        
    def _on_touch_start(self, event):
        """Handle touch start"""
        self.touch_start = (event.x, event.y)
        
    def _on_touch_move(self, event):
        """Handle touch move (for potential gesture detection)"""
        if self.touch_start is None:
            return
            
        # Could add visual feedback here
        pass
        
    def _on_touch_end(self, event):
        """Handle touch end - detect taps and swipes"""
        if self.touch_start is None:
            return
            
        x, y = event.x, event.y
        start_x, start_y = self.touch_start
        
        # Calculate delta
        dx = x - start_x
        dy = y - start_y
        
        # Determine gesture
        if abs(dx) < self.touch_threshold and abs(dy) < self.touch_threshold:
            # Tap detected
            self._handle_tap()
        elif abs(dx) > abs(dy):
            # Horizontal swipe
            if dx > self.touch_threshold:
                self._emit_event('swipe_right')
            else:
                self._emit_event('swipe_left')
        else:
            # Vertical swipe
            if dy > self.touch_threshold:
                self._emit_event('swipe_down')
            else:
                self._emit_event('swipe_up')
                
        self.touch_start = None
        
    def _handle_tap(self):
        """Handle tap (toggle detection)"""
        self._emit_event('toggle')
        
    def _emit_event(self, event_type):
        """Emit an event to the callback"""
        event = {'type': event_type}
        self.event_queue.append(event)
        
    def check_event(self):
        """Check if there's an event in the queue"""
        if hasattr(self, 'event_queue') and self.event_queue:
            return self.event_queue.pop(0)
        return None
        
    def update(self, frame, detections, fps):
        """Update the UI with new frame and detections"""
        self.current_frame = frame
        self.detections = detections
        self.fps = fps
        self.event_queue = []
        
        try:
            # Convert frame to PhotoImage
            if frame is not None:
                # Draw detections on frame
                annotated = self._draw_annotations(frame.copy(), detections)
                
                # Convert to Tkinter format
                display_width = self.canvas.winfo_width() or self.display_config.get('width', 800)
                display_height = self.canvas.winfo_height() or self.display_config.get('height', 480)
                
                # Resize to fit display
                h, w = annotated.shape[:2]
                scale = min(display_width / w, display_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                resized = cv2.resize(annotated, (new_w, new_h))
                
                # Convert color space
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(rgb))
                
                # Update canvas
                self.canvas.delete('all')
                x = (display_width - new_w) // 2
                y = (display_height - new_h) // 2
                self.canvas.create_image(x, y, anchor='nw', image=self.photo)
                
            # Update labels
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.detection_label.config(text=f"Objects: {len(detections)}")
            
            # Schedule next update
            self.root.after(10, self._check_updates)
            
        except Exception as e:
            logger.error(f"UI update error: {e}")
            
    def _check_updates(self):
        """Periodic check for updates"""
        pass  # Updates handled in main loop
        
    def _draw_annotations(self, frame, detections):
        """Draw detection annotations on frame"""
        if not detections:
            return frame
            
        height, width = frame.shape[:2]
        
        for det in detections:
            # Get bounding box
            x1 = int(det['bbox']['x1'] * width)
            y1 = int(det['bbox']['y1'] * height)
            x2 = int(det['bbox']['x2'] * width)
            y2 = int(det['bbox']['y2'] * height)
            
            # Draw box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class']}: {det['confidence']:.0%}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        return frame
        
    def update_model_name(self, name):
        """Update the displayed model name"""
        self.model_label.config(text=f"Model: {name}")
        
    def show_settings(self):
        """Show settings window"""
        if self.settings_window is not None:
            self.settings_window.lift()
            return
            
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Settings")
        self.settings_window.geometry("400x300")
        
        # Settings content
        frame = tk.Frame(self.settings_window, bg=self.colors['bg'])
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Threshold slider
        tk.Label(frame, text="Detection Threshold", bg=self.colors['bg'], 
                fg=self.colors['fg']).pack(anchor='w')
        
        threshold_var = tk.DoubleVar(value=self.detection_config.get('confidence_threshold', 0.5))
        threshold_scale = tk.Scale(
            frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
            variable=threshold_var, resolution=0.1,
            bg=self.colors['bg'], fg=self.colors['fg'],
            highlightthickness=0
        )
        threshold_scale.pack(fill=tk.X, pady=(0, 20))
        
        # Save button
        def save_settings():
            if self.on_settings_change:
                self.on_settings_change({
                    'threshold': threshold_var.get()
                })
            self.settings_window.destroy()
            self.settings_window = None
            
        tk.Button(frame, text="Save", command=save_settings,
                 bg=self.colors['accent'], fg=self.colors['bg']).pack(pady=10)
        
        # Close button
        tk.Button(frame, text="Close", 
                 command=lambda: [self.settings_window.destroy(), 
                                setattr(self, 'settings_window', None)],
                 bg='#666666', fg=self.colors['fg']).pack(pady=5)
        
    def _quit(self):
        """Quit the application"""
        self.running = False
        self._emit_event('quit')
        
    def cleanup(self):
        """Clean up UI resources"""
        self.running = False
        if self.settings_window:
            self.settings_window.destroy()
        self.root.quit()
