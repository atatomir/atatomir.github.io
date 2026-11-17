#!/usr/bin/env python3
"""Static site generator for ScreenToVideo.

All copy lives in this file. Run `python3 _src/build.py` from the s2v folder
(or anywhere) and every page is regenerated as plain HTML next to this folder.
Links are relative, so the output works at a domain root or in a sub-folder.
"""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://screentovideo.com"
MEDIA = SITE + "/media/"
HELP = "https://www.screentovideo.com/help/"
VERSION = ""  # set right after CHANGELOG below, from its newest entry
YEAR = "2026"
COMPANY = "Tatomir N. Alex PFA"
SUPPORT = "support@screentovideo.com"
CONTACT = "contact@screentovideo.com"
INSTALLER = SITE + "/svsetup.exe"
STORE = "https://store.payproglobal.com/checkout?products[1][id]={id}&page-template=13547&products[1][qty]=1"
APPSUMO = "https://appsumo.com/pl/IvhRu9eGae/"
DEFAULT_DESC = "Create amazing videos by adding texts, shapes, zoom effects, sounds and images to your screen recording. The program can be used for creating video tutorials and other educational and training purposes."

e = html.escape


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------
TYPED = ["text animations", "AI blur effects", "transitions", "AI backgrounds", "video effects", "AI voice-overs", "sketches", "your webcam", "Text-to-Speech",
         "cursor effects", "chroma-key", "narration", "3D effects", "interactivity", "animations",
         "system sounds", "annotations", "shapes", "zoom areas", "imported media"]

PILLARS = [
    ("screen", "Record your Screen", "With ScreenToVideo, you can capture an application window, a chosen area, or your entire screen with just one click."),
    ("cloud", "Publish and Share", "Export your MP4 video and share it online via Google Drive. Play on any device with the interactive HTML player."),
    ("sparkles", "Built-in Editor", "Create amazing videos by adding text, shapes, zoom effects, sounds, images, videos, animations, and AI effects to your screen recording."),
    ("zap", "Easy to Use", "ScreenToVideo has an intuitive interface and just the right tools to create a presentation in a matter of minutes!"),
]

# "Record your Screen" card on the home page: each chip shows its own description when clicked
REC_MODES = [
    ("screen", "Full screen", "Capture your entire screen with one click \u2014 ideal for full walkthroughs and presentations."),
    ("media", "Window", "Record a single application window. Fit To Application snaps the capture area to any open app."),
    ("range", "Region", "Drag to select any area, reuse a recent one, or pick a preset size from 240p up to 4K UHD."),
    ("webcam", "Webcam", "Add your webcam with AI background blur or a virtual background \u2014 no green screen needed."),
    ("mic", "Microphone", "Narrate while you record, adjust the volume, and mute or unmute your mic with a hotkey."),
    ("speaker", "System sound", "Capture every sound your computer plays, from app alerts to the audio of a video or game."),
]

HOME_RECORD = ["Record your screen with just one click", "Record your webcam with AI background blur or a virtual background \u2014 no green screen needed",
               "Add voice-over narration at capture time", "Capture any sound played by your computer",
               "Use the countdown timer to get ready for a great start"]
HOME_EDIT = ["Crop, trim and split the screen recording", "Add professional transitions for the entire video or just for an object",
             "Use zoom areas to focus your viewer’s attention", "Add customizable effects to your videos, texts, images, and shapes",
             "Blur or replace the background of any video or image with AI",
             "Bring objects to life with animations", "AI voice-overs with Google Cloud Text-to-Speech (your Google Cloud account) or Microsoft SAPI",
             "Preview all your changes before exporting the video in your favorite format",
             "Export the video to .mp4, .wmv, or .gif file format"]

AUDIENCE = [
    ("briefcase", "Businesses", "Three out of four customers would rather learn about a product or service by watching a video. ScreenToVideo can be used to create engaging interactive presentations that boost conversion and retention rates."),
    ("cap", "Students and Teachers", "Students are more likely to retain a message when watching a video than when reading text. ScreenToVideo provides an efficient way of learning by combining visual and audio information."),
    ("palette", "Hobbyists", "The customizability and ease of use make ScreenToVideo the perfect tool to start video editing. Learn about well-known effects, transitions, and animations while building your own components."),
]

F_RECORD = [
    ("screen", "Screen Recording", "Capture an application window, a chosen area, or your entire screen.", "features_screen_recording"),
    ("webcam", "Webcam Recording", "Record your webcam with AI background blur or a virtual background \u2014 no green screen needed.", "features_webcam_recording"),
    ("mic", "Microphone", "Record your microphone while presenting, or add a voice-over while editing.", "features_microphone"),
    ("speaker", "System Sound", "Capture all sounds played on your computer, e.g. applications and games.", "features_system_sound"),
    ("pen", "Drawing Tools", "Draw, highlight, and write on screen while recording.", "features_drawing_tools"),
    ("camera", "Screenshots", "Take screenshots in the middle of the action.", "features_screenshots"),
    ("sliders", "Adjustment", "Application & Video mode for smooth, high-quality videos.", "features_capture_settings"),
    ("zap", "Quick Results", "Record directly to mp4/gif with no export time.", "features_quick_record"),
]
F_EDIT = [
    ("media", "Import Media", "Import and edit your image, video, and audio files directly in the project.", "features_import_media"),
    ("balloon", "Annotations", "Annotate your presentation with dynamic text enclosed in smooth shapes.", "features_annotations"),
    ("sketch", "Sketch", "Focus attention on specific areas with animated sketches.", "features_sketch"),
    ("sparkles", "Effects", "Highlight or hide information using the built-in effects (blur, pixelate, etc.).", "features_video_effect"),
    ("shapes", "Shapes", "Use geometric shapes to mask objects.", "features_shapes"),
    ("cursor", "Cursor Effects", "Highlight your actions with custom cursor effects.", "features_cursor"),
    ("move", "Zoom and Pan", "Change the perspective of the viewer to an interesting area.", "features_zoom_and_pan"),
    ("zoom", "Zoom Area", "Bring valuable content closer to your viewers’ eyes.", "features_zoom_area"),
    ("sparkles", "AI Background Blur", "Blur the background of any video or image with AI to keep the focus on you.", "features_ai_blur_background"),
    ("person", "AI Virtual Background", "Replace the background of any video or image with AI \u2014 no green screen needed.", "features_ai_virtual_background"),
    ("droplet", "Chroma-Key", "Remove a solid-color background with professional results.", "features_chroma_key_2"),
    ("gauge", "Media Speed", "Speed up or slow down your video or audio.", "features_media_speed"),
    ("brush", "Drawing Tools", "Draw, highlight, and write on images while editing.", "features_drawing"),
    ("silence", "AI Silence Removal", "Automatically find and cut silent pauses in your audio or video.", "features_remove_silences"),
    ("tts", "AI Text-to-Speech", "Turn your script into a natural-sounding voice-over with Google\u2019s neural AI voices (uses your Google Cloud account).", "features_tts"),
    ("narrate", "Narrate", "Enhance the presentation with your voice while editing the video.", "features_narrate"),
    ("loudness", "Loudness Normalization", "Normalized audio from multiple sources fits perfectly into your video.", "features_loudness"),
    ("keyboard", "Keystrokes", "Annotate tutorials with your keystrokes without lifting a finger.", "features_keystroke"),
    ("type", "Text Animations", "Animate your text with a variety of transitions and textures.", "features_text_animation"),
    ("cube", "3D Transformations", "Show your content from a different angle with rotations on all three axes.", "features_transformations"),
    ("transition", "Transitions", "Move from one scene to another with continuous predefined transitions.", "features_video_transition"),
    ("variable", "Text Variables", "Show the elapsed time and other variables in texts.", "features_text_variable"),
    ("copy", "Multiple Selection", "Select, copy, and paste multiple objects with their properties.", "features_multiple_selection"),
    ("layers", "Arrange", "Set the perfect position on both the timeline and the canvas.", "features_arrange"),
    ("range", "Timeline Selection", "Easily select, play, and remove timeline intervals.", "features_timeline_selection"),
    ("motion", "Animations", "Move and transform all objects with smooth custom animations.", "features_animations"),
    ("cc", "Closed Captions", "Create, import, and export closed captions for all your project videos.", "features_closed_captions"),
    ("progress", "Progress Bar", "Illustrate the elapsed time by adding a minimalist progress bar to your videos.", "features_progress_bar"),
    ("click", "Interactivity", "Create buttons and video events to provide more interaction with your viewers.", "features_interactivity"),
]
F_PUBLISH = [
    ("devices", "Export for Any Device", "Choose the perfect export format to be displayed on any device – PC, phone, or tablet."),
    ("code", "HTML Player", "Embed your video in an HTML player that enables interactivity in any browser."),
    ("gif", "Animated GIF", "Create short animated GIFs for your entire video or just a section of it."),
    ("cloud", "Share Online", "Get a shareable link to your videos uploaded directly to Google Drive."),
]
F_EASY = [
    ("eye", "Preview Editor", "Preview all your changes before exporting the video in your favorite format."),
    ("timeline", "Timeline", "Keep track of the precise moments when your objects are displayed."),
    ("inspector", "Object Inspector", "Adjust all properties of your components in a neatly tabbed menu."),
    ("tabs", "Multiple projects", "Work on several projects in parallel without leaving the main application window."),
]

TUTORIALS = [
    ("record_screen", "Record the Screen", "ScreenToVideo manages the whole production cycle: from recording to editing and exporting. This tutorial reveals how simple it is to record your screen along with your webcam and microphone. All input sources come with many adjustable properties."),
    ("recorder_properties", "Setting Recorder Properties", "The recording window has an intuitive design with many options grouped in dropdowns. Set the parameters of the video and start your first project."),
    ("add_text", "Add text", "This video shows how simple it is to create a text area inside a rounded balloon. Both in and out transitions are added to provide a better user experience."),
    ("sketch", "Use a sketch animation", "Sketch animations are very easy to use and can be adjusted to automatically revert and/or continuously loop. Combining them is a good way to draw the attention of your audience."),
    ("texttospeech", "Add an AI voice-over with Text-to-Speech", "Want a professional voice without recording one? Turn your script into a natural-sounding AI voice-over with Google\u2019s neural Text-to-Speech voices (using your Google Cloud account), or choose from the built-in Microsoft voices."),
    ("split", "Add a freeze frame", "Adding freeze frames can be incredibly useful when giving more details. Splitting and removing a segment of your screen recording is also very easy."),
    ("export_player", "Export video with an HTML player", "Exporting has never been so easy. You can embed your video in an HTML player that allows interactivity and choose what happens at the end. Multiple formats are available: MP4, WMV, and GIF."),
    ("interactivity", "Add interactivity to your screen recording", "Let your viewers directly interact with your demo by including buttons, video actions, markers, and a customizable video player."),
]

HELP_TOPICS = [
    ("book", "Getting Started", "getting_started.htm"),
    ("folder", "Creating a new project", "creating_a_new_project.htm"),
    ("webcam", "Recording media", "recording_media.htm"),
    ("film", "Editing a project", "editing_a_project.htm"),
    ("inspector", "Configuring object properties", "configuring_object_properties.htm"),
    ("export", "Exporting your project", "exporting_your_project.htm"),
]

SPECS = [
    ("windows", "Operating System", "Microsoft Windows 11 / 10 / 8 / 7", "Required: 64-bit versions only"),
    ("cpu", "Processor (CPU)", "Intel Core™ i-series or AMD Phenom® II and above", ""),
    ("resolution", "Screen Resolution", "1280 x 720, 32-bit color or above", ""),
    ("disk", "Hard Disk Space", "500 MB for product installation", ""),
    ("memory", "Memory", "Minimum 2 GB RAM", ""),
]

TROUBLE = [
    ("Install Windows 7 Service Pack 1 (SP1)", "Windows 7", "https://support.microsoft.com/en-us/help/15090/windows-7-install-service-pack-1-sp1", "Microsoft Support article"),
    ("Platform update for Windows 7 SP1 and Windows Server 2008 R2 SP1", "Windows 7 and Windows Server 2008 R2", "https://support.microsoft.com/en-us/help/2670838/platform-update-for-windows-7-sp1-and-windows-server-2008-r2-sp1", "Microsoft Support article"),
    ("Media Feature Pack", "Windows 8/8.1/10 N and KN versions", "https://support.microsoft.com/en-gb/help/3145500/media-feature-pack-list-for-windows-n-editions", "Microsoft Support article"),
    ("Install Desktop Experience on Windows Server 2012, 2012 R2", "Windows Server 2012, 2012 R2", "https://youtu.be/FW2gAjKLrfU", "Install Desktop Experience (video)"),
    ("Install Media Foundation on Windows Server 2012, 2012 R2", "Windows Server 2012, 2012 R2", "https://youtu.be/TrBARrUjs1g", "Install Media Foundation (video)"),
    ("Install Server with Desktop Experience", "Windows Server 2016", "https://docs.microsoft.com/en-us/windows-server/get-started/getting-started-with-server-with-desktop-experience", "Microsoft Docs article"),
]

PLANS = [
    dict(key="reg", icon="users", name="Regular license", who="For Individuals and Businesses", base=119, product=57852,
         tiers=[[1, 119], [5, 95], [10, 83], [20, 78], [99, 76]], hot=True),
    dict(key="edu", icon="cap", name="Educational license", who="For Students and Teachers", base=79, product=61278,
         tiers=[[1, 79], [5, 63], [10, 55], [20, 52], [99, 50]], hot=False),
]
PLAN_FEATURES = ["Full-featured recorder and editor", "AI background blur, virtual backgrounds & voice-overs", "Install on multiple devices", "In-app and website support",
                 "Free lifetime upgrades and support", "No monthly/yearly fee is required", "30-day money-back guarantee",
                 "Windows compatibility"]
DISCOUNTS = [(2, 5, "2 – 5 licenses"), (6, 10, "6 – 10 licenses"), (11, 20, "11 – 20 licenses"), (21, 99, "21+ licenses")]


def discount_pct(plan, i):
    # percent saved per license at discount tier i (0-3), from the plan's actual prices
    return round((1 - plan["tiers"][i + 1][1] / plan["base"]) * 100)


def discount_range(i):
    # e.g. "36%" or "36\u201337%" when the two license types differ
    vals = sorted({discount_pct(pl, i) for pl in PLANS})
    return f"{vals[0]}%" if len(vals) == 1 else f"{vals[0]}\u2013{vals[-1]}%"


def max_discount():
    return max(discount_pct(pl, 3) for pl in PLANS)

REDEEM = [  # partner code-redemption pages
    ("appsumo", "AppSumo", "Hey Sumo-Lings!"),
    ("dealmirror", "DealMirror", "Hey!"),
    ("viededingue", "VieDeDingue", "Hey!"),
    ("bypeople", "ByPeople", "Hey ByPeople user!"),
]

UNINSTALL_REASONS = ["I can’t start the program.", "It was not what I expected.", "I don’t like the design.",
                     "The program is hard to use.", "Errors and crashes.", "The program is too expensive.", "Other"]

VP9 = '<a href="https://www.microsoft.com/en-us/p/vp9-video-extensions/9n4d0msmp0pt?activetab=pivot:overviewtab" target="_blank" rel="noopener">VP9 Video Extensions</a>'
HEIF = '<a href="https://www.microsoft.com/en-us/p/heif-image-extensions/9pmmsr1cgpwg?activetab=pivot:overviewtab" target="_blank" rel="noopener">HEIF Image Extensions</a>'
AV1 = '<a href="https://apps.microsoft.com/detail/av1-video-extension/9MVZQVXJBQ9V?hl=en-us" target="_blank" rel="noopener">AV1 Video Extension</a>'


def demo(name):
    return f'(<a href="#" data-video="{MEDIA}{name}.mp4" data-title="Demo">view demo</a>)'


CHANGELOG = f"""
February 11, 2026 | 2026.1
fixed UI scaling issues on dual monitors for a smoother experience
improved AI background blur accuracy and stability
general stability improvements and optimizations
!BETA

May 15, 2025 | 2025.2
added the eyedropper tool
added the crop tool
improved cursor recording
bug fixed: cursor shape recording

January 20, 2025 | 2025.1
added AI blur camera recording
added AI virtual background camera recording
added advanced editing for silence removal {demo('features_remove_silences')}
added chroma-key virtual background in the editor {demo('features_chroma_key_2')}
added AI blur background in the editor for any video/image {demo('features_ai_blur_background')}
added AI virtual background in the editor for any video/image {demo('features_ai_virtual_background')}
improved webcam recording

February 7, 2024 | 2024 (Build 3.1)
added option to exclude certain windows from recording
added encoding and decoding hardware acceleration
added export to .mp4 with VP9 software &amp; hardware encoder (requires {VP9})
added export to .mp4 with AV1 hardware encoder
added export to .wma with Windows Media Audio encoder
added tools to edit video metadata (thumbnail, author, title, description, and copyright)
added import of AV1-encoded .mp4, .webm and .mkv files (requires {AV1})
improved export for Windows Media Video files
bug fixed: incorrectly captured images from some YUY2 webcams
bug fixed: startup error after installing without previously uninstalling the app
bug fixed: missing narration recording from the editor at the first run
bug fixed: incorrectly imported .mp4 file with display aperture

August 21, 2023 | 2023 (Build 3.3)
added option to automatically remove silence from video/audio files
added drag-and-drop import of media files from the explorer
added multi-media import (e.g. images/videos)
added option to replace an image while preserving the object’s properties
updated Google Text-to-Speech voices (including Polyglot voices)
improved the audio recorder
improved timeline smoothness in the Editor
bug fixed: screen recording panel incorrectly displayed for variable DPI
bug fixed: VBR-encoded MP3 files incorrectly imported

April 5, 2023 | 2023 (Build 3.2)
added new shapes/masks in the editor
added ripple delete for single and multiple objects
added timeline multiple selection using the mouse
updated the “Paste properties” form
automatically scale DPI for the camera preview window
improved the timeline horizontal zoom
improved the interface for variable DPI
bug fixed: “Delete Selection” incorrectly disabled
bug fixed: “Split All” incorrectly disabled

February 2, 2023 | 2023 (Build 3.1)
added recording with any display orientation
added the square mask for the webcam in the video recorder
added custom outlines for captions
added text-to-speech for captions
added the square mask for many objects in the editor
added new effects: Emboss, Vignette, and Sepia
updated Google Text-to-Speech voices, including Neural2 voices
bug fixed: magnify/radial blur effect drawn incorrectly when resizing the image/video

October 14, 2022 | 2022 (Build 3.6)
enable/disable hotkeys when recording
center the playhead from the interface
added the hexadecimal color picker
added support for importing and exporting files larger than 4 GB
improved the cursor capture accuracy for variable DPI
improved the quality of imported GIFs with various keyframe sizes
bug fixed: the project name contains illegal characters
bug fixed: importing AVI files in ARGB format (with transparency)
bug fixed: exporting files longer than 50 minutes with changed video speed
bug fixed: showing the wrong email address

June 28, 2022 | 2022 (Build 3.5)
added support for the DirectShow webcam format
added support for the i420 webcam format (DirectShow and Media Foundation)
integrated variables in text areas (e.g. time elapsed/remaining)
added <a href="#" data-video="{MEDIA}features_drawing.mp4" data-title="Animated hand drawings">animated hand drawings</a> on media objects

May 19, 2022 | 2022 (Build 3.4)
added advanced camera settings to improve the video quality
added upload to Google Drive to share videos directly
updated Google Text-to-Speech voices
improved microphone recording flow in the editor

April 5, 2022 | 2022 (Build 3.3)
added the ‘Alignment’ button to the Screen Recorder window
added more freedom when zooming out of the timeline
added option to change the registered license
added option to extend the trial for another 15 days
added property ‘Reflection’ for images and videos
added option to save the reversed video with a customized playback speed
added option to save the video followed by the reversed video with customized playback speeds
save automatically the project’s track height
improved timeline smoothness in the Editor
bug fixed: slow application response after clicking the stop button in the Editor
bug fixed: corrupted audio generated by Google Text-to-Speech on computers with non-English “Language for non-Unicode programs”
bug fixed: sound playing issues when the video/audio is longer than 35 minutes
bug fixed: project exporting issues for videos longer than 35 minutes

February 21, 2022 | 2022 (Build 3.2)
added support for the UYVY webcam format
added multiple capture methods for recording DirectX windows
added customized playback speed in the editor
added customized media speed for audio and video objects
added auto-update button in the interface
improved the quality of imported WebM and MKV files
improved “Split &amp; Freeze” and “Text-to-Speech” tools
improved audio recorder

January 18, 2022 | 2022 (Build 3.1)
quick record to animated WebP format up to 30 frames per second (requires {VP9})
take screenshots in WebP format
adjust the microphone volume in the recording settings
mute/unmute your microphone while recording via hotkey or mouse
import WebM file (VP8/VP9, Opus encoders)
export frames in WebP format
normalize audio perceived loudness per object
export audio objects to file (mp3, wav, or m4a)
export projects as animated WebP files
export project to wav or mp3 file (audio only)
normalize audio perceived loudness at export

November 12, 2021 | 2021 (Build 3.12)
quick record to animated GIF up to 30 frames per second
take screenshots in High Efficiency Image File Format (requires {HEIF})
import WebP short animation (requires {VP9})
support for two new image formats: WebP (.webp) and High Efficiency Image File Format (.heif)
support for animated GIFs with alpha transparency
import screenshots with one click
added new borders and sketches
timeline selection: edit/play/export/delete/ripple delete
added a progress bar for the exported video
export frame in High Efficiency Image File Format
export project to m4a (audio only)
export video with variable frame rate to reduce the file size
added quality and optimization options when exporting to animated and/or transparent GIF
improved audio recorder
fixed the update notification bug

October 19, 2021 | 2021 (Build 3.11)
import Scalable Vector Graphics (SVG) images
create, import, and export closed captions
display keystrokes automatically with the Keystroke object
export Text-to-Speech generated audio files (.wav, .m4a, .mp3)
copy &amp; paste selected properties

September 16, 2021 | 2021 (Build 3.10)
added support for recording a remote desktop connection
added adjustments for the exported playback speed
added support for .mkv files (H264 video encoder and AAC audio encoder)
added the option to change the temporary disk folder for better storage management
enhanced the drawing experience with hotkeys
improved the handling of audio objects in the timeline
fixed Surface Pro camera recording
fixed the timeline interface on 4K displays

June 17, 2021 | 2021 (Build 3.9)
record webcam with virtual background (green screen required)
added two extra cursor effects in the video recorder
added the option to stop a video recording after a preset time
added button shortcuts for trimming and deleting
added button shortcut to focus on the playhead
extended the timeline zoom out up to 25 minutes in Full HD
improved user experience by showing/hiding confirmation messages

May 17, 2021 | 2021 (Build 3.8)
added support for Speech Synthesis Markup Language
added compatibility with virtual cameras
processing multiple video cards
processing long .avi files (longer than 10 minutes)
updated the list of Google TTS voices
extended the Google TTS accents and diacritical marks
improved preview experience for videos over 30 fps
bug fixed: application freezing on exit after repeated video previews

April 6, 2021 | 2021 (Build 3.7)
added support for the NV12 webcam format
extended the zoom out of the timeline
added the timing option: move and extend the time frame
bug fixed: select the correct screen when having access to multiple monitors
bug fixed: application close error after recording the second display

March 25, 2021 | 2021 (Build 3.6)
added support for USB webcams with multiple output streams
added the option to take an extra webcam snapshot
added hardware acceleration when recording the screen and webcam
added object trimming (front/end to playhead)
added project trimming (front/end to playhead)
added the ‘Timeline Tools’ button to easily access the timeline menu
added skipping to the next/previous object
added skipping to the front/end of the object
improved screen recording frame rate on slower computers
bug fixed: not importing/playing audio files with 48 kHz sample rate in Windows 7
bug fixed: the objects created after the Split All operation were in the wrong Z-order

February 25, 2021 | 2021 (Build 3.5)
added support for importing mp4 encoded in HEVC format
added option to create a new project from a video file
improved screen recording quality
higher capture rate on slower computers
improved exported video quality

February 3, 2021 | 2021 (Build 3.3)
screen recording: record to .mp4 video
screen recording: sketch while recording
editor: separate audio and video option
redesigned the screen capture window
fixed minor bugs

December 1, 2020 | 2020 (Build 3.32)
screen recording: added the video capture mode
screen recording: added multiple quality options
webcam recording: added multiple quality options
redesigned the screen capture window
redesigned the intro panel to provide more project details
added support for saving and opening big files
fixed minor bugs

September 14, 2020 | 2020 (Build 3.31)
added the ‘Resize Application’ screen recorder option to fill the capture area
added the text property for an image, video, and screen recording
added a zoom-out option to the Zoom &amp; Pan property
added new shapes, masks, and sketch animations
properties can be displayed in groups (grouped properties)
added helper lines for the timeline
placement can be changed from the keyboard

August 18, 2020 | 2020 (Build 3.30)
added show/hide property on objects
added multi-selection with cut/copy/paste/move/align
added in/out “flip” transition on objects
added “flip” video transition
added object alignment from menu or right panel
added 3D rotations on all axes
added new options for the zoom area object: shadow, border, 3D transform, in/out transitions
extended options for canvas and timeline menu
user interface improvements
fixed minor bugs

July 6, 2020 | 2020 (Build 3.29)
added close button for the active project
user interface improvements
fixed minor bugs

June 9, 2020 | 2020 (Build 3.28)
added tab controllers for each project
user interface improvements
decreased loading time for editing components

April 29, 2020 | 2020 (Build 3.22)
added interactivity property for objects
added video actions on the timeline
added new properties to the sketch animation: loop, revert, and pause
added live chat support
added intro panel with how-to links
added the Split &amp; Freeze option
optimized MP4 video for better performance and streaming
bugs fixed for Windows 7 (service pack 1, media foundation, missing encoders)

March 30, 2020 | 2020 (Build 3.11)
added optional embedding in an HTML video player
added compatibility with all browsers (including mobile devices)
added multiple actions at the beginning and end of each video

February 11, 2020 | 2020 (Build 3.1)
added undo/redo operations
added video transitions: Dissolve, Swirl
added video effect: Grayscale

December 1, 2019 | 2020 (Build 2.21)
added animations
added video transitions: Reveal, Wipe, Slide, Radial Blur, Spin Blur
added H265 support
added video effects: Fisheye Lens

October 1, 2019 | 2020 (Build 1.25)
added Text-to-Speech
added voice recorder
added crop, trim and split recordings
added video effects: Radial Blur, Spin Blur, Swirl
added video transitions: Circle Open, Circle Close, Rectangle Open, Rectangle Close, Pixelate
added zoom and pan

August 1, 2019 | 2020 (Build 1.10)
added texts
added sketch animations
added video effects: Gaussian Blur, Directional Blur, Pixelate
added zoom area
added chroma key effect (green screen)

July 1, 2019 | 2020 (Build 1.0)
first release – record the screen, webcam, microphone, system sound
first release – add shapes, images, videos
first release – export to .mp4 and .wmv
"""

# The site-wide version number is the newest changelog entry, so the two can never disagree
VERSION = CHANGELOG.strip().splitlines()[0].split("|")[1].strip()


FAQ = [
    ("Does ScreenToVideo have AI features?", "Yes. AI background blur and AI virtual backgrounds work while you record your webcam and in the editor, on any video or image \u2014 no green screen needed. AI Text-to-Speech turns your script into a natural-sounding voice-over with Google\u2019s neural voices (using your Google Cloud account) or the built-in Microsoft voices, and AI silence removal cuts pauses from your recordings automatically. Every AI tool is included in the free trial."),
    ("Is there a free trial?", "Yes. You can download a fully featured 15-day free trial. All of ScreenToVideo\u2019s features are available, but your exported videos are watermarked. The projects you make with the trial are fully usable once you register."),
    ("Is it a subscription?", "No. ScreenToVideo is a one-time purchase with no monthly or yearly fee. The license is perpetual and never expires, and it includes free lifetime upgrades and support."),
    ("Can I install it on more than one computer?", "Yes. ScreenToVideo is licensed per user, not per machine, so the registered user can install it on as many computers as they own. Only the registered user may use the software."),
    ("Which formats can I export to?", "MP4, WMV, animated GIF and animated WebP, plus audio-only MP3, WAV and M4A. You can also embed your video in an interactive HTML player or upload it directly to Google Drive and get a shareable link."),
    ("What do I need to run it?", "A 64-bit version of Microsoft Windows 11, 10, 8 or 7, an Intel Core i-series or AMD Phenom II processor, 2 GB of RAM and 500 MB of disk space."),
    ("Are there discounts for schools or teams?", f"Yes. Students and teachers can get the Educational license for $79. Buying several licenses saves you {discount_range(0)} (2\u20135 licenses), {discount_range(1)} (6\u201310), {discount_range(2)} (11\u201320) or {discount_range(3)} (21+)."),
    ("How do I receive my license key?", "Orders are handled securely by PayPro Global. As soon as your order is processed, you receive an email with your registration key, which activates the full version."),
    ("What if it\u2019s not right for me?", "Every purchase comes with a 30-day money-back guarantee."),
    ("Does ScreenToVideo work on Mac?", "No. ScreenToVideo is a Windows app and needs a 64-bit version of Windows 11, 10, 8 or 7."),
    ("Can I record my screen and webcam at the same time?", "Yes. You can record your screen together with your webcam, microphone and system sound, and add AI background blur or a virtual background to the webcam without a green screen."),
    ("Do exported videos have a watermark?", "Only during the free trial. Once you register a license, your videos are exported without a watermark, and projects you made in the trial keep working."),
]


def minify_css(css):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"\s+", " ", css)
    css = re.sub(r"\s*([{};,>])\s*", r"\1", css)
    css = re.sub(r":\s+", ":", css)
    return css.replace(";}", "}").strip()


def build_assets():
    import hashlib
    src = (ROOT / "assets/css/site.css").read_text()
    (ROOT / "assets/css/site.min.css").write_text(minify_css(src))
    h = lambda f: hashlib.sha1((ROOT / f).read_bytes()).hexdigest()[:8]
    return {"css": h("assets/css/site.min.css"), "js": h("assets/js/site.js"), "svg": h("assets/icons.svg")}


ASSET_V = build_assets()


# AppSumo reviews \u2014 copied verbatim from
# https://appsumo.com/products/marketplace-screentovideo/ (update RATING_AS_OF when refreshing).
# "\u2026" marks text left out of a longer review; nothing else is changed.
APPSUMO_PRODUCT = "https://appsumo.com/products/marketplace-screentovideo/"
RATING, RATING_AS_OF = "4.7", "November 2026"
HOME_REVIEWS = 14  # carousel cap; shows 3/2/1 at a time by screen width and loops around
REVIEWS = [
    ("Michael G.", "September 2026", "From purchase to ready in under five minutes",
     "In that short window, I completed the purchase, installed it, launched it, and understood enough of the interface to begin using it. \u2026 ScreenToVideo got me from zero to ready in under five minutes. That is an impressive first experience.",
     "reviews/from-purchase-to-ready-in-under-five-min-393622/"),
    ("faydra_deon", "February 2024", "Yes, they offer the remove silence feature!",
     "I purchased this application back in April 2021, and I literally (no exaggeration) use it daily. \u2026 Thanks for adding the feature. Now I don\u2019t have to switch to another tool for my editing!",
     "reviews/yes-they-offer-the-remove-silence-featu-288746/"),
    ("info7895", "October 2025", "Fantastic recording tool with advanced and easy to use features",
     "This tool has value-to-price ratio perfectly balanced. It provides everything you need for tutorials and talking head videos in an easy-to-use platform. Their tutorials are fantastic and easy to understand. I am so happy to have found this gem on App Sumo. I have other screen recorders, but this one is the best because of its built in editing features.",
     "reviews/fantastic-recording-tool-with-advanced-a-364878/"),
    ("Natasha W.", "April 2026", "Super easy to use",
     "I bought this software because I needed a way to record Zoom calls for workshops. It\u2019s so easy to use, and I see that it also has video editing capability. I haven\u2019t used all the functions yet, but it took me a couple of minutes to set it up and learn how to use it, so I\u2019m sure that this one is a keeper!",
     "reviews/super-easy-to-use-380974/"),
    ("Kira H.", "July 2025", "Easy Peasy Lemon Squeezy",
     "No more OBS Studio for desktop screen recording! Easy to download and use, with quick editing of screen to capture, crop and get creating.",
     "reviews/easy-peasy-lemon-squeezy-356040/"),
    ("gramir", "August 2025", "Great product",
     "It is my goto tool to record and edit videos as it is easy to handle, has a lot of powerful features and is updated on a regular basis.",
     "reviews/great-product-358555/"),
    ("StratosTutorials", "December 2021", "Great Deal and does what it says",
     "It is a stable product that works as expected and does exactly what you are saying. Also love that you are offering one licenses from unlimited personal installation, so I can install it in my 2 computers I have. \u2026 These features would improve the software and add more value to an already great product.",
     "reviews/great-deal-and-does-what-it-says-172001/"),
    ("vsuarez", "November 2023", "Awesome to create video tutorials",
     "Have used some tools for video-recording and this is the most useful and full of features I have seen. It is a pleasure to create video tutorials with lots of effects and control over the final result. Fully recommended!!",
     "reviews/awesome-to-create-video-tutorials-273802/"),
    ("Letsan", "March 2022", "Amazing for video tutorials",
     "This thing records flawlessly. \u2026 And for me the biggest value is in how mouse pointer and clicks are highlighted in quick recording. It is just sooo much better than plain screen recording like other tools. \u2026 the \u201cQuick Recording\u201d option is a godsend, letting me record a compelling, easy to follow tutorial in just minutes.",
     "reviews/amazing-for-video-tutorials-185220/"),
    ("Sumo-linger", "October 2023", "Simple and Super",
     "I\u2019ve been using this for a year now and I\u2019m not a heavy user - It\u2019s a simple tool, but boy is it useful. It does what it says on the tin and it does it well. It\u2019s light weight, so doesn\u2019t use up all of your resources. \u2026 For the price it\u2019s simply amazing value!",
     "reviews/simple-and-super-271248/"),
    ("Verified buyer", "March 2024", "Pretty decent tool",
     "Pretty good tool on the onset with a lightweight (loads fast) and polished feel. Felt the robustness with this tool as recordings were done pretty fast with a fast render time and no glitches and productions were great. Love the cursor effects, \u2018remove silence\u2019 feature, video effects and \u2018zoom and pan\u2019.",
     "reviews/pretty-decent-tool-295825/"),
    ("nedjuric", "March 2021", "Awesome Deal - Great All in One tool",
     "\u2026 in this great offer you get two fantastic products - screen recorder and video editor. In my opinion these apps are complete mature products that contain all the necessary options you need. The support is awesome, authors are willing to accept all ideas and suggestions for new features \u2026",
     "reviews/awesome-deal-great-all-in-one-tool-111303/"),
    ("websam", "June 2022", "Solid Tool for Screen Recording",
     "I\u2019m not sure how I stumbled across this deal but I am glad I did. \u2026 This tool has so many great features. Enough for most people.",
     "reviews/solid-tool-for-screen-recording-198925/"),
    ("TheInfoGuy", "June 2024", "A must needed tool - really excellent recorder.",
     "I used another tool that was not as fast, with limited edit options, etc. This one does a terrific job with plenty of features. Well done!",
     "reviews/a-must-needed-tool-really-excellent-re-307865/"),
]


# --------------------------------------------------------------------------
# Template helpers
# --------------------------------------------------------------------------
class Page:
    def __init__(self, slug):
        self.slug = slug  # '' for home
        self.r = "" if slug == "" else "../" * (slug.count("/") + 1)

    def u(self, path):  # relative url to another page/asset
        return self.r + path


def icon(p, name, cls="i"):
    return f'<svg class="{cls}" aria-hidden="true"><use href="{p.u("assets/icons.svg")}?v={ASSET_V["svg"]}#{name}"/></svg>'


LOGO = ('<svg class="logo-mark" viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="13.6" fill="none" '
        'stroke="#ff5a4e" stroke-width="3.2"/><circle cx="16" cy="16" r="8.2" fill="#ff5a4e"/></svg>'
        '<span class="sr">ScreenToVideo</span><span class="wm" aria-hidden="true"><b>SCREEN</b><i>to</i><b>VIDEO</b></span>')

NAV = [("", "Home"), ("features/", "Features"), ("how-to/", "How-To"), ("support/", "Support"), ("buy/", "Pricing")]


CRUMB_NAMES = {"features": "Features", "how-to": "Video Tutorials", "support": "Support", "buy": "Pricing", "download": "Download",
               "changelog": "Changelog", "contact": "Contact", "system-requirements": "System Requirements",
               "troubleshooting": "Troubleshooting", "eula": "License Agreement", "privacy-policy": "Privacy Policy",
               "website-terms-and-conditions-of-use": "Terms of Use"}


def ld_json(d):
    return f'<script type="application/ld+json">{json.dumps(d, ensure_ascii=False)}</script>'


def ld_breadcrumbs(items):
    # items: [(name, absolute url), ...] starting after Home
    trail = [("Home", SITE + "/")] + items
    return ld_json({"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
        {"@type": "ListItem", "position": i + 1, "name": n, "item": u} for i, (n, u) in enumerate(trail)]})


def layout(p, *, title, desc, content, active=None, page="", track=None, robots=None, extra_body="", nochat=False, canon=None, crumbs=None, ld=""):
    canon = canon or f"{SITE}/{p.slug + '/' if p.slug else ''}"
    if crumbs is None and p.slug in CRUMB_NAMES:
        crumbs = [(CRUMB_NAMES[p.slug], canon)]
    if crumbs and not robots:
        ld += ld_breadcrumbs(crumbs)
    nav = "".join(
        f'<a href="{p.u(h) if h else (p.r or "./")}"{" aria-current=page" if active == h else ""}>{t}</a>' for h, t in NAV)
    attrs = f' data-page="{page}"' + (f' data-track="{track}"' if track else "") + (" data-nochat" if nochat else "")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{e(title)}</title>
<meta name="description" content="{e(desc)}">
<link rel="canonical" href="{canon}">
{f'<meta name="robots" content="{robots}">' if robots else ''}
<meta name="theme-color" content="#0a0d12">
<meta property="og:type" content="website">
<meta property="og:site_name" content="ScreenToVideo">
<meta property="og:title" content="{e(title)}">
<meta property="og:description" content="{e(desc)}">
<meta property="og:url" content="{canon}">
<meta property="og:image" content="{SITE}/assets/img/og.jpg">
<meta property="og:image:alt" content="ScreenToVideo screen recorder and video editor">
<meta property="og:locale" content="en_US">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="{p.u('favicon.svg')}" type="image/svg+xml">
<link rel="icon" href="{p.u('favicon.png')}" sizes="32x32">
<link rel="apple-touch-icon" href="{p.u('apple-touch-icon.png')}">
<link rel="stylesheet" href="{p.u('assets/css/site.min.css')}?v={ASSET_V['css']}">
<script>document.documentElement.classList.add('js')</script>
<script src="{p.u('assets/js/site.js')}?v={ASSET_V['js']}" defer></script>
{ld}
<script type="speculationrules">{{"prefetch":[{{"where":{{"and":[{{"href_matches":"/*"}},{{"not":{{"href_matches":"/*.exe"}}}},{{"not":{{"selector_matches":"[target],[download]"}}}}]}},"eagerness":"moderate"}}]}}</script>
</head>
<body{attrs}>
<a class="skip" href="#main">Skip to content</a>
<header class="hdr">
  <div class="wrap">
    <a class="logo" href="{p.r or './'}" aria-label="ScreenToVideo home">{LOGO}</a>
    <nav class="nav" aria-label="Main">{nav}<a class="btn btn-primary" href="{p.u('download/')}">{icon(p, 'download')}Download Free Trial</a></nav>
    <a class="btn btn-primary btn-sm" href="{p.u('download/')}">{icon(p, 'download')}Download</a>
    <button class="menu-btn" aria-label="Menu" aria-expanded="false" data-icons="{p.u('assets/icons.svg')}?v={ASSET_V['svg']}">{icon(p, 'menu')}</button>
  </div>
</header>
<main id="main">
{content}
</main>
{footer(p)}
{extra_body}
<button class="totop" type="button" aria-label="Back to top">{icon(p, 'arrow', 'i up')}</button>
<div class="consent" role="dialog" aria-label="Cookies">
  <p>We use cookies to improve your experience and measure our ads. See our <a href="{p.u('privacy-policy/')}">Privacy Policy</a>.</p>
  <div class="btns"><button class="btn btn-ghost btn-sm" data-consent="no">Decline</button><button class="btn btn-teal btn-sm" data-consent="yes">Accept</button></div>
</div>
</body>
</html>
"""


def footer(p):
    col = lambda title, links: f'<div><h4>{title}</h4><ul>' + "".join(
        f'<li><a href="{h}"{" target=_blank rel=noopener" if h.startswith("http") else ""}>{t}</a></li>' for h, t in links) + "</ul></div>"
    return f"""<footer class="ftr">
  <div class="wrap">
    <div class="ftr-top">
      <div>
        <a class="logo" href="{p.r or './'}">{LOGO}</a>
        <p>ScreenToVideo is a powerful screen recorder and video editor. Capture an application window, a chosen area, or your entire screen with just one click.</p>
        <div class="social">
          <a href="https://www.facebook.com/sharer/sharer.php?u=https://screentovideo.com/" target="_blank" rel="noopener" aria-label="Share on Facebook">{icon(p, 'facebook')}</a>
          <a href="https://twitter.com/share?url=https://screentovideo.com/" target="_blank" rel="noopener" aria-label="Share on X">{icon(p, 'xlogo')}</a>
          <a href="mailto:{SUPPORT}" aria-label="Email us">{icon(p, 'mail')}</a>
        </div>
      </div>
      {col('Product', [(p.u('features/'), 'Features'), (p.u('how-to/'), 'How-To'), (p.u('buy/'), 'Pricing'), (p.u('download/'), 'Download'), (p.u('changelog/'), 'Changelog')])}
      {col('Help', [(p.u('support/'), 'Support'), (p.u('help/'), 'Help Center'), (p.u('system-requirements/'), 'System Requirements'), (p.u('troubleshooting/'), 'Troubleshooting'), (p.u('contact/'), 'Contact')])}
      {col('Legal', [(p.u('website-terms-and-conditions-of-use/'), 'Terms of Use'), (p.u('privacy-policy/'), 'Privacy Policy'), (p.u('eula/'), 'License Agreement')])}
    </div>
    <div class="ftr-bot">
      <span>&copy; {YEAR} {COMPANY}. All rights reserved.</span>
      <a class="ver" href="{p.u('changelog/')}">Version {VERSION}</a>
    </div>
  </div>
</footer>"""


MODAL = """<dialog class="modal" id="vmodal" aria-label="Video">
  <div class="modal-top"><h3></h3><div class="modal-ctl"><button class="modal-nav" data-dir="-1" aria-label="Previous demo">{prev}</button><button class="modal-nav" data-dir="1" aria-label="Next demo">{next}</button><button class="modal-close" aria-label="Close video">{close}</button></div></div>
  <div class="modal-inner"><video controls playsinline preload="none"></video></div>
</dialog>"""


def modal(p):
    return MODAL.replace("{close}", icon(p, "close")).replace("{prev}", icon(p, "arrow", "i flip")).replace("{next}", icon(p, "arrow"))


def page_hero(p, eyebrow, h1, lead="", center=True, extra=""):
    return f"""<section class="phero{' center' if center else ''}">
  <div class="glow"></div><div class="grid-bg"></div>
  <div class="wrap">
    {f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ''}
    <h1>{h1}</h1>
    {f'<p class="lead">{lead}</p>' if lead else ''}
    {extra}
  </div>
</section>"""


def cta(p, title="Get started with no editing experience!", lead="Fully featured 15-day free trial. Your projects carry over when you register.", second=None):
    second = second or (p.u("features/"), "See All Features")
    return f"""<section class="sec-sm"><div class="wrap">
  <div class="cta center rv">
    <h2>{title}</h2>
    <p class="lead">{lead}</p>
    <div class="btns" style="margin-top:28px">
      <a class="btn btn-primary btn-lg" href="{p.u('download/')}">{icon(p, 'download')}Download Free Trial</a>
      <a class="btn btn-ghost btn-lg" href="{second[0]}">{second[1]}{icon(p, 'arrow')}</a>
    </div>
    <p class="note">{icon(p, 'windows')}Requires Microsoft Windows 7 or newer (64-bit)</p>
  </div>
</div></section>"""


def audience(p):
    cards = "".join(
        f'<details class="card card-spot rv aud-item" style="--rd:{i * .08}s" open><summary><span class="num">0{i + 1}</span><div class="ic-box{" c" if i == 1 else ""}">{icon(p, ic)}</div><h3>{t}</h3></summary><p>{d}</p></details>'
        for i, (ic, t, d) in enumerate(AUDIENCE))
    return f"""<section class="sec aud"><div class="wrap">
  <div class="head center rv"><div class="eyebrow">Made for everyone</div><h2>Who is ScreenToVideo designed for?</h2></div>
  <div class="grid g3">{cards}</div>
</div></section>"""


def checks(items, c=False):
    return f'<ul class="checks{" c" if c else ""}">' + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def write(slug, content):
    out = ROOT / slug / "index.html" if slug else ROOT / "index.html"
    if slug == "404":
        out = ROOT / "404.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def home():
    p = Page("")
    words = "|".join(TYPED)
    viz = [
        '<div class="viz chips" role="group" aria-label="Recording modes">' + "".join(
            f'<button type="button" class="chip{" on" if i == 0 else ""}" aria-pressed="{"true" if i == 0 else "false"}" data-desc="{e(dsc)}">{icon(p, ic)}{t}</button>'
            for i, (ic, t, dsc) in enumerate(REC_MODES)) + '</div>',
        '<div class="viz fmt"><span>.mp4</span><span>.gif</span><span>.wmv</span><span>HTML</span><span>Drive</span></div>',
        '<div class="viz wave" aria-hidden="true">' + "".join(
            f'<i style="--h:{h}%;--d:-{i * .09:.2f}s"></i>' for i, h in enumerate([30, 55, 80, 45, 95, 60, 35, 70, 100, 50, 30, 65, 85, 40, 25, 60, 90, 55, 35, 75, 50, 30])) + '</div>',
        '<div class="viz tl" aria-hidden="true"><div style="--l:0%;--w:62%"></div><div style="--l:18%;--w:40%;--c:var(--coral)"></div><div style="--l:40%;--w:45%;--c:#7f9cff"></div><div style="--l:8%;--w:30%"></div></div>',
    ]
    order = [0, 1, 2, 3]
    bento = "".join(
        f'<article class="card card-spot rv" style="--rd:{k * .06}s"><div class="ic-box{" c" if k % 3 == 0 else ""}">{icon(p, PILLARS[i][0])}</div><h3>{PILLARS[i][1]}</h3>'
        + (f'<p class="rec-desc" aria-live="polite">{REC_MODES[0][2]}</p>' if i == 0 else f'<p>{PILLARS[i][2]}</p>') + f'{viz[i]}</article>'
        for k, i in enumerate(order))
    content = f"""
<section class="hero center">
  <div class="glow"></div><div class="grid-bg"></div>
  <div class="wrap">
    <a class="pill rv" href="{p.u('changelog/')}"><b>New</b><span><span class="hide-sm">Version {VERSION}: </span>Improved AI background blur</span>{icon(p, 'arrow')}</a>
    <h1 class="rv" style="--rd:.05s">Create Video Tutorials <span class="typed-line">with <span class="typed" data-words="{e(words)}">{TYPED[0]}</span><span class="caret" aria-hidden="true"></span></span></h1>
    <p class="lead rv" style="--rd:.1s">The most customizable screen recorder and video editor, with AI background blur, virtual backgrounds, voice-overs and silence removal. Create impressive presentations from scratch.</p>
    <div class="btns rv" style="--rd:.15s">
      <a class="btn btn-primary btn-lg" href="{p.u('download/')}">{icon(p, 'download')}Download Free Trial</a>
      <a class="btn btn-ghost btn-lg" href="{p.u('features/')}">See All Features{icon(p, 'arrow')}</a>
    </div>
    <p class="note rv" style="--rd:.2s">{icon(p, 'check')}Fully featured 15-day free trial. Requires Windows 7 or newer (64-bit).</p>
    <div class="shot rv" style="--rd:.25s">
      <div class="shot-media">
        <img src="assets/img/hero-editor.webp" width="1280" height="720" alt="The ScreenToVideo editor: adding a text balloon to a project, with the properties panel and timeline" fetchpriority="high">
        <button class="watch" data-video="{MEDIA}home_editor_2.mp4" data-title="ScreenToVideo Editor"><span>{icon(p, 'play')}</span>Watch overview</button>
      </div>
    </div>
    <div class="strip rv">
      <div>{icon(p, 'clock')}<span><b>15-day free trial</b>All features included</span></div>
      <div>{icon(p, 'infinity')}<span><b>Lifetime license</b>No monthly/yearly fee</span></div>
      <div>{icon(p, 'refresh')}<span><b>Free upgrades</b>Lifetime updates &amp; support</span></div>
      <div>{icon(p, 'shield')}<span><b>30-day guarantee</b>Money-back, no questions</span></div>
    </div>
  </div>
</section>

<section class="sec"><div class="wrap">
  <div class="split">
    <div class="split-media rv">
      <img src="assets/img/home-recorder.webp" width="960" height="540" alt="ScreenToVideo screen recorder quick setup window" loading="lazy" decoding="async">
      <button class="play play-sm" data-video="{MEDIA}home_screen_recorder_2.mp4" data-title="Screen Recorder" aria-label="Play the screen recorder video"><span class="play-btn">{icon(p, 'play')}</span></button>
    </div>
    <div class="rv" style="--rd:.1s">
      <div class="eyebrow">Screen recorder</div>
      <h2>Record your screen in minutes</h2>
      {checks(HOME_RECORD)}
      <a class="btn btn-ghost" href="{p.u('features/#recording')}">Learn more{icon(p, 'arrow')}</a>
    </div>
  </div>
</div></section>

<section class="sec" style="padding-top:0"><div class="wrap">
  <div class="split rev">
    <div class="split-media rv">
      <img src="assets/img/home-editor.webp" width="960" height="540" alt="ScreenToVideo editor adding an annotation balloon" loading="lazy" decoding="async">
      <button class="play play-sm" data-video="{MEDIA}home_editor_2.mp4" data-title="Video Editor" aria-label="Play the video editor video"><span class="play-btn">{icon(p, 'play')}</span></button>
    </div>
    <div class="rv" style="--rd:.1s">
      <div class="eyebrow">Video editor</div>
      <h2>All editing tools in one place</h2>
      {checks(HOME_EDIT, c=True)}
      <a class="btn btn-ghost" href="{p.u('features/#editing')}">Learn more{icon(p, 'arrow')}</a>
    </div>
  </div>
</div></section>

{ai_section(p)}
{reviews(p)}
<section class="sec"><div class="wrap">
  <div class="head center rv"><div class="eyebrow">Everything in one app</div><h2>Powerful <span class="grad">Screen Recorder and Video Editor</span></h2></div>
  <div class="bento">{bento}</div>
</div></section>

{audience(p)}
{pricing_band(p)}
{faq(p)}
{cta(p)}
"""
    content += ld_home()
    return write("", layout(p, title="ScreenToVideo: AI Screen Recorder & Video Editor for Windows",
                            desc="Screen recorder and video editor for Windows with AI background blur, virtual backgrounds and voice-overs. Create video tutorials fast. Free 15-day trial.",
                            content=content, active="", page="home", track="home", extra_body=modal(p)))


def stars(p, n=5):
    return '<span class="stars" aria-hidden="true">' + "".join(icon(p, "star") for _ in range(n)) + "</span>"


def rating_stars(p):
    # Gold stars clipped to the exact rating (4.7 -> four full stars + 70% of the fifth) over dim ones.
    # Star 17px, gap 2px (see .stars in site.css).
    r = float(RATING)
    full = int(r)
    px = full * 19 + (r - full) * 17
    return (f'<span class="stars-rate" role="img" aria-label="{RATING} out of 5 stars">'
            f'<span class="stars stars-off" aria-hidden="true">{"".join(icon(p, "star") for _ in range(5))}</span>'
            f'<span class="stars stars-on" aria-hidden="true" style="width:{px:.1f}px">{"".join(icon(p, "star") for _ in range(5))}</span></span>')


AI_CARDS = [
    ("sparkles", "AI Background Blur", "Blur a messy room behind you \u2014 while recording your webcam, or later in the editor on any video or image.", "features_ai_blur_background"),
    ("person", "AI Virtual Background", "Replace the background behind you, while recording your webcam or later on any video or image. No green screen needed.", "features_ai_virtual_background"),
    ("tts", "AI Voice-Overs", "Type your script and get a natural-sounding narration from Google\u2019s neural voices (with your Google Cloud account) or the built-in Microsoft voices.", "features_tts"),
    ("silence", "AI Silence Removal", "Find and cut the pauses in your recording automatically, so your video gets to the point.", "features_remove_silences"),
]


def ai_section(p):
    cards = "".join(ftile(p, ic, t, d, v, i % 2 == 1, i * .05, badge=False) for i, (ic, t, d, v) in enumerate(AI_CARDS))
    # phones: the same four features as tap-to-open rows, each with its demo button
    rows = "".join(
        f'<details class="card ai-item"><summary><span class="ic-box{" c" if i % 2 else ""}">{icon(p, ic)}</span><h3>{t}</h3></summary>'
        f'<div class="ai-body"><p>{d}</p><button type="button" class="btn btn-ghost btn-sm ai-demo" data-video="{MEDIA}{v}.mp4" data-title="{e(t)}">{icon(p, "play")}Watch demo</button></div></details>'
        for i, (ic, t, d, v) in enumerate(AI_CARDS))
    return f'''<section class="sec"><div class="wrap">
  <div class="head center rv"><div class="eyebrow">AI built in</div><h2>Let AI do the <span class="grad">tedious work</span></h2>
  <p class="lead">Professional-looking videos without a studio: AI cleans up your background, narrates your script and trims the dead air.</p></div>
  <div class="grid g4 ai-grid">{cards}</div>
  <div class="ai-acc">{rows}</div>
</div></section>'''


def reviews(p):
    cards = "".join(
        f'''<figure class="rvw" role="group" aria-roledescription="review" aria-label="{i + 1} of {min(HOME_REVIEWS, len(REVIEWS))}">
  <div class="rev-top">{stars(p)}<span class="sr">5 out of 5 stars</span></div>
  <h3>{e(t)}</h3>
  <blockquote><p>\u201c{e(q)}\u201d</p></blockquote>
  <figcaption><span class="av" aria-hidden="true">{e(n[0].upper())}</span><span class="rev-date">{d} \u00b7 AppSumo</span>
  <a class="rev-link" href="{APPSUMO_PRODUCT}{u}" target="_blank" rel="noopener" aria-label="Read the full review on AppSumo">{icon(p, "external")}</a></figcaption>
</figure>''' for i, (n, d, t, q, u) in enumerate(REVIEWS[:HOME_REVIEWS]))
    return f'''<section class="sec"><div class="wrap">
  <div class="rev-head rv">
    <div>
      <div class="eyebrow">Customer reviews</div>
      <h2>Loved by creators on <span class="grad">AppSumo</span></h2>
    </div>
    <a class="score" href="{APPSUMO_PRODUCT}#reviews" target="_blank" rel="noopener">
      <b>{RATING}</b>
      <span>{rating_stars(p)}<small>on AppSumo</small></span>
    </a>
  </div>
  <div class="revs rv" tabindex="0" aria-label="Customer reviews" aria-roledescription="carousel">{cards}</div>
  <div class="rev-nav rv">
    <a class="btn btn-ghost" href="{APPSUMO_PRODUCT}#reviews" target="_blank" rel="noopener">Read more reviews on AppSumo{icon(p, "external")}</a>
    <div class="rev-ctl">
      <button type="button" class="rev-btn rev-prev" aria-label="Previous reviews">{icon(p, "arrow", "i back")}</button>
      <button type="button" class="rev-btn rev-next" aria-label="Next reviews">{icon(p, "arrow")}</button>
    </div>
  </div>
</div></section>'''


def pricing_band(p):
    def mini(name, who, price, hot):
        return (f'<div class="mini{" hot" if hot else ""}"><div><b>{name}</b><span>{who}</span></div>'
                f'<div class="mini-price"><b>${price}</b><span>one-time</span></div></div>')
    return f"""<section class="sec-sm"><div class="wrap">
  <div class="band rv">
    <div>
      <div class="eyebrow">Pricing</div>
      <h2>Pay once. <span class="grad">Own it forever.</span></h2>
      <p class="lead">No subscription. Free lifetime upgrades and support, multi-license discounts up to {max_discount()}%, and a 30-day money-back guarantee.</p>
      <div class="btns" style="margin-top:24px"><a class="btn btn-primary" href="{p.u('buy/')}">See pricing{icon(p, 'arrow')}</a><a class="btn btn-ghost" href="{p.u('download/')}">Try free for 15 days</a></div>
    </div>
    <div class="minis">{mini("Regular license", "For Individuals and Businesses", 119, True)}{mini("Educational license", "For Students and Teachers", 79, False)}</div>
  </div>
</div></section>"""


def faq(p, keys=None, title="Questions, answered"):
    sel = [f for i, f in enumerate(FAQ) if keys is None or i in keys]
    items = "".join(f'<details class="qa rv"><summary>{e(q)}<span class="pm" aria-hidden="true"></span></summary><p>{e(a)}</p></details>' for q, a in sel)
    return f"""<section class="sec"><div class="wrap">
  <div class="faq">
    <div class="rv"><div class="eyebrow">FAQ</div><h2>{title}</h2><p class="muted">Can\u2019t find what you\u2019re looking for? <a href="{p.u('contact/')}">Contact us</a> \u2014 we reply within 24 hours.</p></div>
    <div class="qas">{items}</div>
  </div>
</div></section>"""


COMPARE = [
    ("All recording, editing and AI features", True, True),
    ("Export to MP4, WMV, GIF, WebP & HTML player", True, True),
    ("Upload and share via Google Drive", True, True),
    ("Videos without watermark", False, True),
    ("Use beyond 15 days", False, True),
    ("Free lifetime upgrades and support", False, True),
]


def compare(p):
    yes = icon(p, "check")
    dash = "\u2014"
    rows = "".join(f'<div class="cmp-row"><span>{t}</span><span class="{"y" if a else "n"}">{yes if a else dash}</span><span class="y">{yes if b else dash}</span></div>' for t, a, b in COMPARE)
    return f"""<section class="sec-sm"><div class="wrap narrow">
  <div class="head center rv"><div class="eyebrow">Trial vs. full version</div><h2>Try everything first</h2><p class="lead">The free trial includes every feature. Projects you make with it stay fully usable after you register.</p></div>
  <div class="cmp rv">
    <div class="cmp-row cmp-head"><span></span><span>Free trial</span><span>Full license</span></div>
    {rows}
    <div class="cmp-row cmp-foot"><span></span><span><a class="btn btn-ghost btn-sm" href="{p.u('download/')}">Download</a></span><span><a class="btn btn-primary btn-sm" href="#pricing-top">Buy now</a></span></div>
  </div>
</div></section>"""


def ld_faq(keys=None):
    sel = [f for i, f in enumerate(FAQ) if keys is None or i in keys]
    return ld_json({"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
        {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in sel]})


def ld_home():
    data = [
        {"@context": "https://schema.org", "@type": "SoftwareApplication", "@id": SITE + "/#app", "name": "ScreenToVideo",
         "operatingSystem": "Windows 11, Windows 10, Windows 8, Windows 7 (64-bit)", "applicationCategory": "MultimediaApplication",
         "applicationSubCategory": "Screen recorder and video editor",
         "softwareVersion": VERSION, "url": SITE + "/", "downloadUrl": INSTALLER, "installUrl": SITE + "/download/",
         "image": SITE + "/assets/img/og.jpg", "screenshot": SITE + "/assets/img/hero-editor.webp",
         "description": "Screen recorder and video editor for Windows with AI background blur, AI virtual backgrounds, AI voice-overs and AI silence removal. Record your screen, webcam, microphone and system sound, then add text, shapes, zoom effects, transitions and animations, and export to MP4, GIF, WMV or an interactive HTML player.",
         "softwareRequirements": "64-bit Windows 11, 10, 8 or 7; Intel Core i-series or AMD Phenom II; 2 GB RAM; 500 MB disk space",
         "featureList": [t for _, t, *_ in F_RECORD + F_EDIT + F_PUBLISH + F_EASY],
         "publisher": {"@id": SITE + "/#org"},
         "offers": [{"@type": "Offer", "name": "Regular license", "price": "119", "priceCurrency": "USD", "url": SITE + "/buy/", "availability": "https://schema.org/InStock"},
                    {"@type": "Offer", "name": "Educational license", "price": "79", "priceCurrency": "USD", "url": SITE + "/buy/", "availability": "https://schema.org/InStock"},
                    {"@type": "Offer", "name": "15-day free trial", "price": "0", "priceCurrency": "USD", "url": SITE + "/download/"}]},
        {"@context": "https://schema.org", "@type": "Organization", "@id": SITE + "/#org", "name": "ScreenToVideo", "legalName": COMPANY,
         "url": SITE + "/", "logo": SITE + "/apple-touch-icon.png", "email": SUPPORT, "foundingDate": "2019"},
        {"@context": "https://schema.org", "@type": "WebSite", "@id": SITE + "/#site", "name": "ScreenToVideo", "url": SITE + "/", "publisher": {"@id": SITE + "/#org"}},
        {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in FAQ]},
    ]
    return "".join(f'<script type="application/ld+json">{json.dumps(d, ensure_ascii=False)}</script>' for d in data)


AI_TILES = {"Webcam Recording", "AI Background Blur", "AI Virtual Background", "AI Text-to-Speech", "AI Silence Removal"}


def ftile(p, ic, t, d, v=None, c=False, delay=0, badge=True):
    title = t  # plain name, used for the video popup title
    if badge and t in AI_TILES:
        t = f'{t}<span class="ai-badge">AI</span>' if not t.startswith("AI ") else f'{t[3:]}<span class="ai-badge">AI</span>'
    if v:
        return (f'<button class="card card-spot ftile rv" style="--rd:{delay}s" data-video="{MEDIA}{v}.mp4" data-title="{e(title)}">'
                f'<span class="ic-box{" c" if c else ""}">{icon(p, ic)}</span><h3>{t}</h3><p>{d}</p>'
                f'<span class="demo">{icon(p, "play")}View demo</span></button>')
    return f'<article class="card card-spot rv" style="--rd:{delay}s"><span class="ic-box{" c" if c else ""}">{icon(p, ic)}</span><h3>{t}</h3><p>{d}</p></article>'


def features():
    p = Page("features")
    groups = [
        ("recording", "Recording", "Start recording with just one click", F_RECORD, True, False, "g4"),
        ("editing", "Editing", "Edit a video with powerful tools", F_EDIT, True, True, "g4"),
        ("publishing", "Publishing", "Publish and share", F_PUBLISH, False, False, "g4"),
        ("easy", "Workflow", "Easy to use", F_EASY, False, True, "g4"),
    ]
    sub = "".join(f'<a href="#{gid}">{name}</a>' for gid, name, *_ in groups)
    body = ""
    for gid, name, title, items, vids, c, g in groups:
        tiles = "".join(ftile(p, ic, t, d, (rest[0] if vids else None), c, (i % 4) * .05) for i, (ic, t, d, *rest) in enumerate(items))
        body += f"""<section class="fgroup" id="{gid}"><div class="wrap">
  <div class="fgroup-head rv"><div><div class="eyebrow">{name}</div><h2>{title}</h2></div><span class="count">{len(items):02d} features</span></div>
  <div class="grid {g}">{tiles}</div>
</div></section>"""
    content = page_hero(p, "Features", 'Everything you need to <span class="grad">record, edit &amp; share</span>',
                        "A great screen recorder and an easy-to-use video editor in one app. Click any feature to watch a short demo.",
                        extra=f'<div class="btns" style="margin-top:28px"><a class="btn btn-primary btn-lg" href="{p.u("download/")}">{icon(p, "download")}Download Free Trial</a><a class="btn btn-ghost btn-lg" href="{p.u("how-to/")}">Watch tutorials{icon(p, "arrow")}</a></div>')
    n = sum(len(g[3]) for g in groups)
    search = (f'<label class="fsearch">{icon(p, "zoom")}<span class="sr">Search features</span>'
              f'<input type="search" placeholder="Search {n} features" autocomplete="off" data-fsearch></label>')
    content += f'<nav class="subnav" aria-label="Feature categories"><div class="wrap"><div class="subnav-links">{sub}</div>{search}</div></nav>' + body
    content += '<div class="wrap"><p class="fempty" hidden>No features match your search. <a href="#" data-fclear>Clear search</a></p></div>'
    content += cta(p, second=(p.u("how-to/"), "Watch tutorials"))
    return write("features", layout(p, title="Screen Recorder & Video Editor Features - ScreenToVideo",
                                    desc=f"All {len(F_RECORD) + len(F_EDIT) + len(F_PUBLISH) + len(F_EASY)} ScreenToVideo features: screen and webcam recording, AI background blur and voice-overs, annotations, zoom, effects and export to MP4, GIF and more.",
                                    content=content, active="features/", page="features", track="features", extra_body=modal(p)))


def tut_cards(p, keys=None):
    items = [t for t in TUTORIALS if keys is None or t[0] in keys]
    return "".join(
        f'<button class="card tut rv" style="--rd:{(i % 2) * .06}s" data-video="{MEDIA}tutorial_{k}_2.mp4" data-title="{e(t)}">'
        f'<span class="thumb"><img src="{p.u(f"assets/img/tut-{k}.webp")}" width="640" height="360" alt="" loading="lazy" decoding="async">'
        f'<span class="play play-sm" aria-hidden="true"><span class="play-btn">{icon(p, "play")}</span></span></span>'
        f'<span class="body"><h3>{t}</h3><p>{d}</p></span></button>'
        for i, (k, t, d) in enumerate(items))


def howto():
    p = Page("how-to")
    content = page_hero(p, "How-To", "Learn ScreenToVideo <span class=\"grad\">in minutes</span>",
                        "Short video tutorials that walk you through recording, editing and exporting your first project.")
    content += f'<section class="sec-sm" style="padding-top:0"><div class="wrap"><h2 class="sr">Video tutorials</h2><div class="grid g2">{tut_cards(p)}</div></div></section>'
    content += cta(p, "Try the full-featured trial version for free", "or see more demos on the features page!")
    return write("how-to", layout(p, title="Video Tutorials - ScreenToVideo", desc="Short ScreenToVideo video tutorials: record your screen, add text and sketch animations, use AI text-to-speech, add freeze frames and export with an HTML player.",
                                  content=content, active="how-to/", page="howto", track="howto", extra_body=modal(p)))


def install():
    p = Page("install")
    content = page_hero(p, "Installation complete", "Thank you for installing <span class=\"grad\">ScreenToVideo!</span>",
                        "Below you can find some short tutorials on how to create your video in a matter of minutes.")
    content += f'<section class="sec-sm" style="padding-top:0"><div class="wrap"><h2 class="sr">Quick tutorials</h2><div class="grid g3">{tut_cards(p, ["record_screen", "recorder_properties", "export_player"])}</div>'
    content += f'<div class="btns rv" style="justify-content:center;margin-top:40px"><a class="btn btn-primary" href="{p.u("features/")}">See All Features{icon(p, "arrow")}</a><a class="btn btn-ghost" href="{p.u("how-to/")}">More tutorials</a></div></div></section>'
    return write("install", layout(p, title="Thank you for installing ScreenToVideo!", desc="ScreenToVideo is installed. Watch three short tutorials to record your screen, set up the recorder and export your first video in minutes.",
                                   content=content, page="install", track="install", extra_body=modal(p), robots="noindex"))


def support():
    p = Page("support")
    cards = "".join(
        f'<a class="card card-spot ftile rv" style="--rd:{(i % 3) * .05}s" href="{p.u('help/' + f)}"><span class="ic-box{" c" if i % 2 else ""}">{icon(p, ic)}</span><h3>{t}</h3><p>{help_covers(f)}</p><span class="demo">{icon(p, "arrow")}Read the guide</span></a>'
        for i, (ic, t, f) in enumerate(HELP_TOPICS))
    content = page_hero(p, "Support", "Welcome to the ScreenToVideo <span class=\"grad\">support page</span>",
                        "If you would like to know more about how to use our program, feel free to take a look at our help center below.")
    content += f"""<section class="sec-sm" style="padding-top:0"><div class="wrap"><h2 class="sr">Help topics</h2>
  <div class="grid g3">{cards}</div>
  <div class="grid g3" style="margin-top:16px">
    <a class="card card-spot rv" href="mailto:{SUPPORT}"><span class="ic-box c">{icon(p, 'mail')}</span><h3>Email support</h3><p>If you cannot find the answers you are looking for, please don’t hesitate to contact us at <span class="link">{SUPPORT}</span> and we will get back to you within 24 hours.</p></a>
    <a class="card card-spot rv" style="--rd:.05s" href="{p.u('troubleshooting/')}"><span class="ic-box">{icon(p, 'wrench')}</span><h3>Troubleshooting</h3><p>Install the Windows components (service packs, Media Feature Pack, Media Foundation) required for recording and playback.</p></a>
    <a class="card card-spot rv" style="--rd:.1s" href="{p.u('how-to/')}"><span class="ic-box">{icon(p, 'play')}</span><h3>Video tutorials</h3><p>Watch short tutorials on recording, editing, adding text-to-speech, and exporting your project.</p></a>
  </div>
</div></section>"""
    return write("support", layout(p, title="Support - ScreenToVideo", desc="ScreenToVideo support: step-by-step help guides, video tutorials, troubleshooting for Windows components and email support with replies within 24 hours.",
                                   content=content, active="support/", page="support", track="support"))


def buy():
    p = Page("buy")
    plans = ""
    for pl in PLANS:
        rows = '<div><span>Multi-license discounts</span><span></span></div>' + "".join(
            f'<div data-min="{mn}" data-max="{mx}"><span>{lbl}</span><b>{discount_pct(pl, i)}% off</b></div>' for i, (mn, mx, lbl) in enumerate(DISCOUNTS))
        plans += f"""<article class="card plan{' hot' if pl['hot'] else ''} rv" data-plan="{pl['key']}" data-base="{pl['base']}" data-product="{pl['product']}" data-tiers="{e(json.dumps([[1, pl['base']]] + pl['tiers'][1:]))}">
  {'<span class="badge">Most popular</span>' if pl['hot'] else ''}
  <span class="ic-box{' c' if pl['hot'] else ''}">{icon(p, pl['icon'])}</span>
  <h3>{pl['name']}</h3><p class="for">{pl['who']}</p>
  <div class="price"><b>${pl['base']}</b><span>one-time fee</span></div>
  {checks(PLAN_FEATURES, c=pl['hot']).replace('<ul class="checks', '<ul class="checks dup', 1) if not pl['hot'] else checks(PLAN_FEATURES, c=True)}
  {'' if pl['hot'] else '<p class="same-feat">Includes every feature of the Regular license.</p>'}
  <div class="tiers">{rows}</div>
  <div class="qty"><label for="q-{pl['key']}">Licenses</label>
    <div class="stepper"><button type="button" data-step="-1" aria-label="Fewer licenses">−</button><input id="q-{pl['key']}" type="number" inputmode="numeric" min="1" max="99" value="1"><button type="button" data-step="1" aria-label="More licenses">+</button></div>
    <div class="unit"><b>${pl['base']}</b> / license</div>
  </div>
  <a class="btn {'btn-primary' if pl['hot'] else 'btn-teal'} btn-lg btn-block buy" href="{STORE.format(id=pl['product'])}"><span>Buy Now (${pl['base']})</span>{icon(p, 'arrow')}</a>
</article>"""
    infos = [
        ("shield", "Secure Payments", ["Orders are handled by the PayPro Global payment processing service. PayPro Global is an electronic payment processing service that guarantees secure transfer and processing of your payment information."]),
        ("key", "The License", ["As soon as your order is processed, you will receive an email containing the registration key. The key is used to activate the full version of the software.",
                                "The license is perpetual and never expires, so you are legally entitled to use the full version of ScreenToVideo as long as you need it.",
                                "ScreenToVideo is licensed on a per-user basis, which means that it is licensed to the registered user, not the machine it is installed on. This allows the registered user to install the software on as many machines as they own, but only the registered user may use the software."]),
        ("refresh", "Updates", ["Your purchase of ScreenToVideo includes FREE lifetime upgrades and support."]),
    ]
    info = "".join(f'<article class="card rv" style="--rd:{i * .06}s"><span class="ic-box{" c" if i == 1 else ""}">{icon(p, ic)}</span><h3>{t}</h3>' + "".join(f"<p>{x}</p>" for x in ps) + "</article>" for i, (ic, t, ps) in enumerate(infos))
    content = page_hero(p, "Pricing", "Get a <span class=\"grad\">lifetime license</span>",
                        "Pay once, own it forever. Free lifetime upgrades and support, with a 30-day money-back guarantee.")
    content += f"""<section class="sec-sm" style="padding-top:0"><div class="wrap"><h2 class="sr">Licenses</h2>
  <div style="max-width:960px;margin:0 auto">
    <div class="coupon" id="coupon">
      <p>Use coupon code <code id="couponcode" data-copy title="Click to copy">SV50OFF</code> at checkout to get <em>50% OFF!</em></p>
      <a class="btn btn-primary" id="btncode" href="{STORE.format(id=57852)}&coupon-code-to-add=SV50OFF"><span>Buy Now ($59.50)</span>{icon(p, 'arrow')}</a>
    </div>
    <a class="sumo rv" href="{APPSUMO}" target="_blank" rel="sponsored noopener"><b>AppSumo</b><em class="sumo-rate">{icon(p, 'star')}{RATING}</em> Record, edit, and share videos <span>Get the deal</span>{icon(p, 'external')}</a>
  </div>
  <div class="plans" id="pricing-top">{plans}</div>
  <div class="grid g3 infos" style="margin-top:clamp(56px,8vw,96px)">{info}</div>
</div></section>
{compare(p)}
{faq(p, [2, 3, 6, 7, 8], "Buying questions")}
{ld_faq([2, 3, 6, 7, 8])}
{cta(p, "Not sure yet? Try it free for 15 days.", "All features are available in the trial — your projects stay fully usable after you register.", (p.u('features/'), 'See All Features'))}"""
    return write("buy", layout(p, title="Pricing: Lifetime License, No Subscription - ScreenToVideo",
                               desc="Buy ScreenToVideo once and own it forever: $119 regular or $79 educational license, free lifetime upgrades, multi-license discounts and a 30-day money-back guarantee.",
                               content=content, active="buy/", page="buy", track="pricing"))


def download():
    p = Page("download")
    content = f"""<section class="phero center">
  <div class="glow"></div><div class="grid-bg"></div>
  <div class="wrap">
    <div class="eyebrow">Download</div>
    <h1>Thanks for choosing <span class="grad">ScreenToVideo</span></h1>
    <p class="lead">Your 15-day free trial is on its way.</p>
    <div class="dl" style="margin-top:36px;text-align:left">
      <div class="spinner">{icon(p, 'download')}</div>
      <div><h2 id="dlstate">Downloading, please wait…</h2><p>If your download has not started automatically, please <a href="{INSTALLER}">click here</a>.</p></div>
    </div>
    <div class="dl dl-alt" hidden style="margin-top:36px;text-align:left">
      <div class="spinner" style="background:rgba(31,191,174,.14)">{icon(p, 'windows')}</div>
      <div>
        <h2>ScreenToVideo is a Windows app</h2>
        <p>It looks like you\u2019re not on a Windows PC. Copy the link and open it on your Windows PC, or download the installer anyway.</p>
        <div class="btns" style="margin-top:16px">
          <a class="btn btn-primary btn-sm" href="{INSTALLER}">{icon(p, 'download')}Download anyway</a>
          <button class="btn btn-ghost btn-sm" type="button" data-copylink="{SITE}/download/">{icon(p, 'copy')}<span>Copy link</span></button>
        </div>
      </div>
    </div>
  </div>
</section>
<section class="sec-sm" style="padding-top:0"><div class="wrap">
  <div class="grid g2" style="max-width:960px;margin:0 auto">
    <article class="card rv"><span class="ic-box c">{icon(p, 'clock')}</span><h3>15-day Free Trial Version</h3><p>All of ScreenToVideo’s features are available, but your exported videos are watermarked.</p><p style="margin-top:10px">The projects that you have made with a trial version of ScreenToVideo will be fully usable with a registered version.</p></article>
    <article class="card rv" style="--rd:.05s"><span class="ic-box">{icon(p, 'download')}</span><h3>Install ScreenToVideo</h3><ol class="steps" style="margin-top:14px"><li>When prompted, save the file to a convenient location on your computer.</li><li>Open the file, then the setup program will guide you through the rest of the process.</li></ol></article>
    <article class="card rv"><span class="ic-box">{icon(p, 'close')}</span><h3>Uninstall ScreenToVideo</h3><ol class="steps" style="margin-top:14px"><li>Ensure ScreenToVideo and its help window are closed.</li><li>Click the Windows Start button.</li><li>Select Control Panel (or Settings → Apps on Windows 10/11).</li><li>Double-click Add or Remove Programs (Programs and Features).</li><li>Select ScreenToVideo, click the Remove button, and click Yes.</li></ol></article>
    <article class="card rv" style="--rd:.05s"><span class="ic-box c">{icon(p, 'book')}</span><h3>Help center</h3><p>If you would like to know more about how to use our program, feel free to take a look at our <a href="{p.u('how-to/')}">how-to guide</a> or <a href="{p.u('help/')}">help center</a>.</p><p style="margin-top:10px">Check the <a href="{p.u('system-requirements/')}">system requirements</a> or the <a href="{p.u('troubleshooting/')}">troubleshooting page</a> if anything doesn’t work as expected.</p></article>
  </div>
</div></section>"""
    return write("download", layout(p, title="Download the Free Trial for Windows - ScreenToVideo",
                                    desc="Download the free 15-day trial of ScreenToVideo, the screen recorder and video editor with AI background blur and voice-overs, for Windows 11, 10, 8 and 7.",
                                    content=content, page="download", track="download"))


def classify(item):
    s = re.sub("<[^>]+>", "", item).lower()
    if s.startswith(("bug fixed", "bugs fixed", "fixed", "solved")):
        return "fix", "Fixed"
    if s.startswith(("improved", "updated", "enhanced", "optimized", "redesigned", "extended", "decreased", "higher", "general", "user interface")):
        return "imp", "Improved"
    return "new", "New"


def changelog_blocks():
    return [b.strip().splitlines() for b in CHANGELOG.strip().split("\n\n")]


def changelog():
    p = Page("changelog")
    blocks = changelog_blocks()
    out, year, years, counts = "", None, [], {"new": 0, "imp": 0, "fix": 0}
    for idx, b in enumerate(blocks):
        date, ver = [x.strip() for x in b[0].split("|")]
        y = date[-4:]
        if y != year:
            out += ("</section>" if year else "") + f'<section class="cl-ysec" id="y{y}"><div class="cl-year rv" aria-hidden="true">{y}</div>'
            year = y
            years.append(y)
        items, beta = "", ""
        for it in b[1:]:
            if it == "!BETA":  # the beta installer it pointed to no longer exists (404), so the note is not shown
                continue
            cls, lbl = classify(it)
            counts[cls] += 1
            txt = re.sub(r"^(added |bug fixed: )", "", it)
            txt = txt[0].upper() + txt[1:] if not txt.startswith("<") else txt
            items += f'<li data-k="{cls}"><span class="tagl t-{cls}">{lbl}</span><span>{txt}</span></li>'
        slug = re.sub(r"[^a-z0-9]+", "-", ver.lower()).strip("-")
        out += f"""<article class="cl-item{' latest' if idx == 0 else ''} rv" id="v{slug}">
  <div class="cl-meta"><time>{date}</time><span class="cl-ver">v{ver}</span></div>
  <span class="cl-dot" aria-hidden="true"></span>
  <div class="cl-body"><h2 class="sr">Version {ver}</h2><ul>{items}</ul>{beta}</div>
</article>"""
    content = page_hero(p, "Changelog", "What’s <span class=\"grad\">new</span>",
                        f"Every ScreenToVideo release since 2019. Your license includes all of them — free lifetime upgrades.")
    out += "</section>"
    total = sum(counts.values())
    chips = "".join(
        f'<button type="button" class="clf{" on" if k == "all" else ""}" data-k="{k}" aria-pressed="{"true" if k == "all" else "false"}">{lbl} <span>{n}</span></button>'
        for k, lbl, n in [("all", "All", total), ("new", "New", counts["new"]), ("imp", "Improved", counts["imp"]), ("fix", "Fixed", counts["fix"])])
    yl = "".join(f'<a href="#y{y}">{y}</a>' for y in years)
    content += (f'<nav class="subnav" aria-label="Changelog years and filters"><div class="wrap"><div class="subnav-links">{yl}</div>'
                f'<div class="clfs" role="group" aria-label="Filter changes">{chips}</div></div></nav>')
    content += f'<section class="sec-sm"><div class="wrap narrow"><p class="cl-sum muted">{len(blocks)} releases \u00b7 {total} changes since July 2019</p><div class="cl">{out}</div></div></section>'
    return write("changelog", layout(p, title="Changelog & Release Notes - ScreenToVideo",
                                     desc="Every ScreenToVideo release since 2019: new features, improvements and fixes, including AI background blur, AI virtual backgrounds and silence removal.",
                                     content=content, page="changelog", track="changelog", extra_body=modal(p)))


def sysreq():
    p = Page("system-requirements")
    rows = "".join(
        f'<article class="card spec rv" style="--rd:{i * .05}s"><span class="ic-box{" c" if i == 0 else ""}">{icon(p, ic)}</span><h3>{t}</h3><p>{v}{f"<small>{n}</small>" if n else ""}</p></article>'
        for i, (ic, t, v, n) in enumerate(SPECS))
    content = page_hero(p, "System requirements", "Runs on almost any <span class=\"grad\">Windows PC</span>",
                        "ScreenToVideo is lightweight. Here’s what you need to record, edit and export.")
    content += f"""<section class="sec-sm" style="padding-top:0"><div class="wrap narrow"><h2 class="sr">Requirements</h2>
  <div class="specs">{rows}</div>
  <div class="callout rv" style="margin-top:20px">{icon(p, 'wrench')}<div><p>Using Windows 7, an N/KN edition or Windows Server? You may need extra Windows components. See <a href="{p.u('troubleshooting/')}">Troubleshooting</a>.</p></div></div>
</div></section>
{cta(p)}"""
    return write("system-requirements", layout(p, title="System Requirements - ScreenToVideo", desc="ScreenToVideo system requirements: 64-bit Windows 11, 10, 8 or 7, an Intel Core i-series or AMD Phenom II CPU, 2 GB of RAM and 500 MB of disk space.", content=content, page="sysreq"))


def troubleshooting():
    p = Page("troubleshooting")
    rows = "".join(
        f'<a class="card card-spot ftile rv" style="--rd:{(i % 2) * .05}s" href="{u}" target="_blank" rel="noopener"><span class="ic-box{" c" if "youtu" in u else ""}">{icon(p, "play" if "youtu" in u else "windows")}</span><h3>{t}</h3><p>Applies to: {a}</p><span class="demo">{icon(p, "external")}{lbl}</span></a>'
        for i, (t, a, u, lbl) in enumerate(TROUBLE))
    content = page_hero(p, "Troubleshooting", "Install the required <span class=\"grad\">Windows components</span>",
                        "We use Windows components to provide the best integration. This page helps you install the required packs containing video encoders, decoders and players.")
    content += f"""<section class="sec-sm" style="padding-top:0"><div class="wrap"><h2 class="sr">Windows components</h2>
  <div class="grid g2">{rows}</div>
  <div class="callout c rv" style="margin-top:20px">{icon(p, 'chat')}<div><p>If you need more help, do not hesitate to <a href="mailto:{SUPPORT}">contact our support team</a> or start a live chat.</p></div></div>
</div></section>"""
    return write("troubleshooting", layout(p, title="Troubleshooting Windows Components - ScreenToVideo",
                                           desc="Install the Windows components ScreenToVideo needs: Windows 7 SP1, the Media Feature Pack for N/KN editions, and Media Foundation for Windows Server.",
                                           content=content, page="troubleshooting"))


def field(name, label, typ="text", req=True, auto="", full=False):
    r = " required" if req else ""
    star = " <i>*</i>" if req else ""
    a = f' autocomplete="{auto}"' if auto else ""
    fid = "f-" + re.sub(r"[^a-z0-9]+", "-", name.lower())
    if typ == "textarea":
        return f'<div class="field"><label for="{fid}">{label}{star}</label><textarea class="input" id="{fid}" name="{name}"{r}></textarea></div>'
    return f'<div class="field"><label for="{fid}">{label}{star}</label><input class="input" id="{fid}" name="{name}" type="{typ}"{a}{r}></div>'


HONEYPOT = '<div class="hp" aria-hidden="true"><input name="_gotcha" tabindex="-1" autocomplete="off"></div>'


def contact():
    p = Page("contact")
    opts = "".join(f'<label class="opt"><input type="radio" name="Subject" value="{s}"{" required" if i == 0 else ""}><span>{s}</span></label>' for i, s in enumerate(["Bug report", "Request new feature", "License issue", "Other"]))
    content = page_hero(p, "Contact", "How can we <span class=\"grad\">help?</span>", "Feel free to ask a question or simply leave a comment. We usually reply within 24 hours.")
    content += f"""<section class="sec-sm" style="padding-top:0"><div class="wrap">
  <div class="contact-grid">
    <form class="card panel form rv" data-form data-subject="ScreenToVideo contact">
      <div><h2 style="font-size:1.3rem;margin:0 0 4px">Your Details</h2><p class="muted" style="margin:0">Let us know how to get back to you.</p></div>
      <div class="row2">{field('First Name', 'First Name', auto='given-name')}{field('Last Name', 'Last Name', auto='family-name')}</div>
      {field('Email', 'Email Address', 'email', auto='email')}
      <div class="field"><span class="legend">Subject <i>*</i></span><div class="opts" role="radiogroup">{opts}</div></div>
      {field('Message', 'Comments / Questions', 'textarea')}
      {HONEYPOT}
      <div class="form-msg" role="status"></div>
      <div><button class="btn btn-primary btn-lg" type="submit">Send message{icon(p, 'arrow')}</button></div>
    </form>
    <div class="grid" style="gap:16px">
      <article class="card rv" style="--rd:.05s"><span class="ic-box c">{icon(p, 'mail')}</span><h3>Email</h3><p>Support: <a href="mailto:{SUPPORT}">{SUPPORT}</a></p><p style="margin-top:6px">Careers &amp; partnerships: <a href="mailto:{CONTACT}">{CONTACT}</a></p></article>
      <article class="card rv" style="--rd:.1s"><span class="ic-box">{icon(p, 'users')}</span><h3>About us</h3>
        <p>Founded in 2019, we are an information technology company developing screen recording and video editing software for home and office.</p>
        <p style="margin-top:10px">Our developers have 20+ years of experience in this area and want to build the complete toolbox for amateurs and professional editors alike. That is the reason behind our main product – ScreenToVideo.</p>
        <p style="margin-top:10px">After an incredible growth year, our team renamed the company to {COMPANY} in 2025. If you want to be part of our adventure, please <a href="mailto:{CONTACT}">contact us</a>.</p></article>
    </div>
  </div>
</div></section>"""
    return write("contact", layout(p, title="Contact Us - ScreenToVideo",
                                   desc="Contact the ScreenToVideo team with a question, bug report, license issue or feature request. We usually reply within 24 hours.",
                                   content=content, page="contact"))


def uninstall():
    p = Page("uninstall")
    opts = "".join(f'<label class="opt"><input type="checkbox" name="Reason" value="{e(s)}"><span>{e(s)}</span></label>' for s in UNINSTALL_REASONS)
    content = page_hero(p, "Uninstalled", "We are sorry to see <span class=\"grad\">you go!</span>",
                        "You have successfully uninstalled ScreenToVideo. Please let us know why — it helps us make the program better.")
    content += f"""<section class="sec-sm" style="padding-top:0"><div class="wrap narrow">
  <form class="card panel form rv" data-form data-subject="ScreenToVideo uninstall feedback" data-ok="Thank you for your feedback!">
    <div class="field"><span class="legend">Please let us know why you are uninstalling ScreenToVideo:</span><div class="opts">{opts}</div></div>
    {field('Message', 'Comments / Questions', 'textarea', req=False)}
    <div class="row2">{field('First Name', 'First Name', req=False, auto='given-name')}{field('Last Name', 'Last Name', req=False, auto='family-name')}</div>
    {field('Email', 'Email Address', 'email', req=False, auto='email')}
    {HONEYPOT}
    <div class="form-msg" role="status"></div>
    <div><button class="btn btn-primary btn-lg" type="submit">Send feedback{icon(p, 'arrow')}</button></div>
  </form>
</div></section>"""
    return write("uninstall", layout(p, title="ScreenToVideo - Uninstall", desc="ScreenToVideo has been uninstalled. Tell us why so we can make the screen recorder and video editor better.", content=content, page="uninstall", track="uninstall", robots="noindex"))


def redeem(slug, partner, hello):
    p = Page(slug)
    content = page_hero(p, f"{partner} customers", f"{e(hello)}",
                        f"Please enter your name, email, and {partner} code to get your registration details.")
    content += f"""<section class="sec-sm" style="padding-top:0"><div class="wrap narrow">
  <form class="card panel form rv" data-redeem="{SITE}/license/{slug}.php" data-partner="{partner}">
    <div class="row2">{field('first', 'First Name', auto='given-name')}{field('last', 'Last Name', auto='family-name')}</div>
    {field('email', 'Email Address', 'email', auto='email')}
    {field('code', f'{partner} code', auto='off')}
    {HONEYPOT}
    <div class="form-msg" role="status"></div>
    <div class="redeem-result" aria-live="polite" hidden></div>
    <div><button class="btn btn-primary btn-lg" type="submit">Get my registration details{icon(p, 'arrow')}</button></div>
  </form>
</div></section>"""
    return write(slug, layout(p, title=f"Redeem Your {partner} Code - ScreenToVideo", desc="Redeem your partner code to get your ScreenToVideo registration details by email.", content=content, page="redeem", robots="noindex"))


def gdrive():
    p = Page("uploadtogoogledrive")
    content = f"""<section class="phero center"><div class="glow"></div><div class="grid-bg"></div><div class="wrap">
  <div class="card status">
    <div data-status="authorized"><div class="status-ic ok-ic">{icon(p, 'check')}</div><h1 style="font-size:1.9rem">You have successfully connected to your Google Drive.</h1>
      <p class="muted">Please return to the application to upload and share your videos.</p>
      <p class="muted">You can always see or remove access in your Google Account: <a href="https://myaccount.google.com/permissions" target="_blank" rel="noopener">myaccount.google.com/permissions</a></p></div>
    <div data-status="denied"><div class="status-ic no-ic">{icon(p, 'close')}</div><h1 style="font-size:1.9rem">The authorization process was unsuccessful.</h1>
      <p class="muted">You won’t be able to upload videos to Google Drive. Please return to the application and try again.</p></div>
    <div data-status="error"><div class="status-ic no-ic">{icon(p, 'alert')}</div><h1 style="font-size:1.9rem">Something went wrong</h1>
      <p class="muted">An internal error occurred while connecting to your Google Drive. Please return to the application and try again.</p>
      <p class="muted">If the problem persists, please contact us at <a href="mailto:{CONTACT}">{CONTACT}</a>.</p></div>
    <noscript><p>Please return to the application.</p></noscript>
    <p class="note" style="justify-content:center">{icon(p, 'cloud')}Upload to Google Drive</p>
  </div>
</div></section>"""
    return write("uploadtogoogledrive", layout(p, title="ScreenToVideo - Upload to Google Drive", desc="Connect ScreenToVideo to Google Drive to upload your videos and share them with a link.", content=content,
                                               page="gdrive", track="uploadtogd", robots="noindex", nochat=True))


def legal(slug, title, h1, desc, sections, intro="", eyebrow="Legal"):
    p = Page(slug)
    toc = "".join(f'<a href="#s{i}">{re.sub("<[^>]+>", "", h)}</a>' for i, (h, _) in enumerate(sections) if h)
    body = intro + "".join((f'<h2 id="s{i}">{h}</h2>' if h else "") + "".join(f"<p>{x}</p>" for x in ps) for i, (h, ps) in enumerate(sections))
    content = page_hero(p, eyebrow, h1, center=False)
    content += f'<section class="sec-sm" style="padding-top:0"><div class="wrap"><div class="legal"><nav class="toc" aria-label="On this page"><b>On this page</b>{toc}</nav><div class="prose">{body}</div></div></div></section>'
    return write(slug, layout(p, title=title, desc=desc, content=content, page="legal"))


def eula():
    S = [
        ("1. Registered version", ["A single registered copy of ScreenToVideo may either be used by a single person who uses the software himself or herself on one or more computers, or installed on a single computer and used by multiple people, but not both.",
                                   "You may access the registered version of ScreenToVideo through a network, provided that you have obtained individual licenses for the software covering all computers that will run the software from the network. For instance, if seven different workstations will access ScreenToVideo on the network, you will need a seven-user ScreenToVideo license, whether they use ScreenToVideo at different times or concurrently.",
                                   "With a registered copy of ScreenToVideo, you are able to create an unlimited number of freely distributable, royalty-free video files."]),
        ("2. Shareware version / evaluation and registration", ["This is not free software. Subject to the terms below, you are hereby licensed to use this software for evaluation purposes without charge for a period of 15 days. If you use this software after the 15-day evaluation period, a registration fee is required for the Full Version."]),
        ("3. Copyright", ["The Software is confidential copyrighted information of Aladni Software. You shall not modify, decompile, disassemble, decrypt, extract, or otherwise reverse engineer software. The Software may not be leased, assigned, or sublicensed, in whole or in part. The commercial version of ScreenToVideo may either be used by a single person who uses the Software personally on one or more computers, or installed on a single workstation used non-simultaneously by multiple people, but not both. You may access a version of ScreenToVideo through a network, provided that you have obtained individual commercial licenses for the Software covering all workstations that will access the Software through the network."]),
        ("4. Limitation of support", ["If you are using the Software free of charge under the terms of this agreement, you are not entitled to support or telephone assistance."]),
        ("5. Multi-user packages, company licenses", ["If the copy of the Software you received was accompanied by a printed or other form of “hard-copy” License whose terms vary from this agreement, then the hard-copy License governs your use of the Software."]),
        ("6. Disclaimer of warranty", ['<span class="caps">THIS SOFTWARE AND THE ACCOMPANYING FILES ARE SOLD “AS IS” AND WITHOUT WARRANTIES AS TO PERFORMANCE OR MERCHANTABILITY OR ANY OTHER WARRANTIES WHETHER EXPRESSED OR IMPLIED. Because of the various hardware and software environments into which ScreenToVideo may be put, NO WARRANTY OF FITNESS FOR A PARTICULAR PURPOSE IS OFFERED.</span>',
                                       "A good data processing procedure dictates that any program be thoroughly tested with non-critical data before relying on it.",
                                       "The user must assume the entire risk of using the program.",
                                       '<span class="caps">ANY LIABILITY OF THE SELLER WILL BE LIMITED EXCLUSIVELY TO PRODUCT REPLACEMENT OR REFUND OF PURCHASE PRICE.</span>']),
        ("7. High-risk activities", ["The Software is not fault-tolerant and is not designed, manufactured, or intended for use or resale as online control equipment in hazardous environments requiring fail-safe performance, such as in the operation of nuclear facilities, air traffic control, aircraft navigation or communication systems, direct life support machines, or weapons systems, in which the failure of the Software could lead directly to death, personal injury, or severe physical or environmental damage (“High-Risk Activities”). Aladni Software specifically disclaims any express or implied warranty of fitness for High-Risk Activities."]),
        ("8. Distribution of evaluation version", ["Provided that you verify that you are distributing the evaluation version, you are hereby licensed to make as many copies of the evaluation version of this software and documentation as you wish; give exact copies of the original evaluation version to anyone, and distribute the evaluation version of the software and documentation in its unmodified form via electronic means. There is no charge for any of the above.",
                                                   "You are specifically prohibited from charging, or requesting donations, for any such copies, however made; and from distributing the software and/or documentation with other products (commercial or otherwise) without prior written permission."]),
    ]
    intro = "<p><strong>You should carefully read the following terms and conditions before using this software.</strong> Unless you have a different license agreement signed by Aladni Software, the use or installation of this software indicates your acceptance of this license agreement and warranty.</p>"
    return legal("eula", "License Agreement (EULA) - ScreenToVideo", "License Agreement", "The ScreenToVideo end-user license agreement: registered and evaluation versions, copyright, support, multi-user licenses and warranty.", S, intro)


def privacy():
    S = [
        ("1. What data do we collect and how do we use it?", [
            "<strong>Information you provide to us.</strong> When you participate in, access, purchase or sign up to any of our services, activities, or online content, such as subscriptions, newsletters, surveys, competitions, or events, we receive personal information about you which we use to be able to provide these services. This may consist of data including your name, email address, postal address, and telephone number. You will be able to select how we use your information for marketing purposes and you can update your marketing preferences at any time.",
            "<strong>Account information.</strong> Some of our online services (like file sharing and subscription for software products) require a password-protected account. We collect and associate with your account the information you provide to us when you do activities such as signing up for your account or making purchases for our Services (like your full name, email address, and billing address).",
            "<strong>Usage information.</strong> We collect information related to how you use the Services, including actions you take when using our Services (like editing, viewing, and sharing Your Stuff, changing themes, activating software license, and logging in and out of your account). We use this information to improve our Services, develop new services and features, and protect you as well as other ScreenToVideo users.",
            "<strong>Device information.</strong> We also collect information from and about the devices you use to access the Services. This includes things like IP addresses, system language, the type of browser and device you use, the version of our software, the web page you visited before coming to our sites, and identifiers associated with your devices. Your devices (depending on their settings) may also transmit location information to the Services.",
            "<strong>Crash reports and diagnostic information.</strong> We may also collect diagnostic reports to help troubleshoot software bugs, fix significant errors, and improve overall user experience. These reports may include things like file names, hardware model, process information, app, and device information. To collect this information, we will ask for your consent in advance.",
            "<strong>Cookies and other technologies.</strong> We use technologies like cookies to provide, improve, protect, and promote our Services. For example, cookies help us with things like remembering your username for your next visit, understanding how you are interacting with our Services, and improving them based on that information. You can set your browser to not accept cookies, but this may limit your ability to use the Services.",
            "<strong>Marketing.</strong> We do not send you any marketing material unless you voluntarily subscribe to our newsletters (opt-in). We collect your email address and other information you submit when you subscribe. If you do not want to receive marketing materials from us anymore, simply click the ‘Unsubscribe’ link in any email at any time (opt-out)."]),
        ("2. Legally process personal data", [
            "We will not share your personal data with other parties except as provided below:",
            "2.1. We may share personal information with Service Providers. We provide certain Services and Software through Service Providers. We may share your personal identification information with such Service Providers to fulfill orders, deliver packages, send mail or email, administer contests or sweepstakes, manage customer lists, analyze data, provide marketing assistance, process credit card and other payments, operate the Website, troubleshoot, and provide customer service. We will also share your information with Service Providers to accomplish our administrative tasks.",
            "2.2. We may disclose the User’s personal data if ScreenToVideo is under a duty to disclose or share such personal data in order to comply with any legal obligation, or in order to protect the rights, property, or safety of its business, its customers or others.",
            "2.3. We may also transmit your personal information to companies that help us to run our business by processing personal information on behalf of us for the purposes identified above. Such companies include providers of payment processing services, data analysis services providers, fraud monitoring and prevention providers, email delivery service providers, social media, and other marketing platforms and service providers.",
            "2.4. The collected personal data may be transferred by ScreenToVideo to recipients that are located outside the European Union and such destinations may not have laws that protect the User’s personal data to the same extent as in the European Union. ScreenToVideo ensures that the User’s personal data processed by ScreenToVideo or by its suppliers and partners operating outside the European Union are treated securely and are protected against unauthorized access, loss or destruction, unlawful processing, and any processing which is inconsistent with the purposes set out in this privacy policy.",
            "2.5. We may use third-party tracking tools (such as Adwords, Bing, Criteo, fabric, Facebook, Firebase, Flurry, Google Analytics, Hotjar, LinkedIn, Pardot, Quora, JPUSH) to track the information on how users use the products and websites; such information will be stored on third-party servers. Such third-party tracking tools may send the data analysis report for ScreenToVideo’s above-mentioned purpose.",
            "2.6. We may store the collected information on the servers leased from providers of third-party server services.",
            "2.7. <strong>Google Ads.</strong> Our website uses the Google Ads tag, provided by Google, to measure how our ads perform \u2014 for example, when someone who clicked one of our ads downloads ScreenToVideo. Until you click \u201cAccept\u201d in our cookie banner, the tag runs in Google Consent Mode with ad storage, ad user data, ad personalization and analytics storage set to \u201cdenied\u201d: no advertising cookies are stored, and Google receives only limited signals without cookies. If you accept, Google may set cookies and use them to measure conversions and to personalize ads. You can change your choice at any time by clearing this website\u2019s data in your browser, after which the cookie banner is shown again. To learn how Google uses this information, see the <a href=\"https://policies.google.com/privacy\" target=\"_blank\" rel=\"noopener\">Google Privacy Policy</a> and <a href=\"https://policies.google.com/technologies/partner-sites\" target=\"_blank\" rel=\"noopener\">How Google uses information from sites that use its services</a>.",
            "2.8. <strong>Tawk.to live chat.</strong> Our website includes a live chat widget provided by tawk.to inc., so you can contact us directly from any page. The widget loads on our pages and may set cookies and process technical information about your visit, such as your IP address, browser, and the pages you view, in order to provide the chat. If you start a chat, the messages and any details you enter, such as your name or email address, are processed and stored by tawk.to so that we can reply to you. For more information, see the <a href=\"https://www.tawk.to/privacy-policy/\" target=\"_blank\" rel=\"noopener\">tawk.to Privacy Policy</a>."]),
        ("3. Storage and retention of data", [
            "<strong>User Controls.</strong> You can access, amend, download, and delete your personal information by logging into your ScreenToVideo account and going to your account settings page.",
            "<strong>Retention.</strong> When you sign up for an account with us, we’ll retain information associated with your account for as long as your account is in existence or as long as we need it to provide you the Services. If you delete your account, we will initiate the deletion of this information immediately. But please note: (1) there might be some latency in deleting this information from our servers and back-up storage; and (2) we may retain this information if necessary to comply with our legal obligations, resolve disputes, or enforce our agreements. Other information will be retained until you ask us to delete it."]),
        ("4. Security", ["We are committed to protecting the personal data we hold. In accordance with GDPR, we adopt appropriate technical and organizational measures to help prevent unauthorized access to your information and ensure data is held securely. We also require any third parties to whom we may transfer personal data to have appropriate security measures in place. We ask third parties to complete Data Security Questionnaires so we can be sure they are a safe place for your data."]),
        ("5. Your right to control and access your information", [
            "You have control over your personal information and how it is collected, used, and shared. For example, you have a right to:",
            "5.1. Access and take your data. You can download copies of Your Stuff from our website. You can also ask us for a copy of the personal data you provided to us.",
            "5.2. Change or correct personal data. You can manage your account and the content contained in it, as well as edit some of your profile details, through your account settings page.",
            "5.3. Erase or delete all or some of Your Stuff in your ScreenToVideo account. You can delete each of the files you uploaded in your account, through your ‘My Maps’ page.",
            "5.4. Delete your entire account. You can delete your ScreenToVideo account, as well as any information related to it once and for all, through your account settings page.",
            "5.5. Unsubscribe from our newsletters. Simply click any ‘Unsubscribe’ link in any marketing email you received to have your email address deleted from our mailing list.",
            "5.6. Your right to withdraw consent. Whenever you have given us your consent to use your personal data, you have the right to change your mind at any time and withdraw that consent.",
            "5.7. The User has the right to define general or specific instructions regarding storage, deletion, and use of their personal data after death."]),
        ("6. Policy changes", ["If we are involved in a reorganization, merger, acquisition, or sale of our assets, your information may be transferred as part of that deal. We will notify you (for example, via a message to the email address associated with your account) of any such deal and outline your choices in that event. We may revise this Privacy Policy from time to time and will post the most current version on our website. If a revision meaningfully reduces your rights, we will notify you."]),
        ("7. Contact us", [f'Have questions or concerns about ScreenToVideo, our Services, and privacy? Contact our Data Protection Officer at <a href="mailto:{SUPPORT}">{SUPPORT}</a>. If they can’t answer your question, you have the right to contact your local data protection supervisory authority.']),
    ]
    intro = ("<p>ScreenToVideo (collectively referred to as “we”, “us”, and “our”) fully respects your privacy. This policy covers the processing of personal data by ScreenToVideo, which is committed to protecting the privacy of your personal information whilst striving to provide the very best user experience. We want our websites to be safe and enjoyable for everyone.</p>"
             "<p>The operations of ScreenToVideo are in accordance with the European Union’s General Data Protection Regulation (GDPR), effective May 25, 2018. ScreenToVideo has made the GDPR a priority, and we are and have always been fully aligned with the regulation’s intended result: the protection of your privacy and personal data.</p>")
    return legal("privacy-policy", "Privacy Policy - ScreenToVideo", "Privacy Policy", "How ScreenToVideo collects, uses, stores and protects your personal data under the GDPR, and how you can access, change or delete it.", S, intro)


def terms():
    S = [
        ("Copyright", ["This website and its materials (including, but not limited to, text, graphics, logos, sounds, and software) are the copyrighted material of ScreenToVideo. All rights reserved.",
                       "No part of this publication may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of ScreenToVideo excluding the following situations:",
                       "1. You may print or download extracts to a local hard disk for your personal and non-commercial use only.",
                       "2. You may copy the content to individual third parties for their personal use, but only if you acknowledge the website as the source of the material."]),
        ("Indemnity", ["You hereby indemnify to the fullest extent ScreenToVideo from and against any and all liabilities, costs, demands, causes of action, damages, and expenses (including reasonable attorney’s fees) arising out of or in any way related to your breach of any of the provisions of these Terms."]),
        ("Disclaimer", ["Although every precaution has been taken in advance in order to verify the accuracy of the information contained herein, neither we nor any third parties provide any warranty or guarantee the accuracy, timeliness, performance, completeness, or suitability of the information, materials, and software found or offered on this website for any particular purpose. You acknowledge that such information and materials may contain inaccuracies or errors and we expressly exclude liability for any of such inaccuracies or errors to the fullest extent permitted by law.",
                        "Your use of any information, materials, or software on this website is entirely at your own risk. It shall be your own responsibility to ensure that any products, services, or information available through this website meet your specific requirements."]),
        ("Links to Other Websites", ["Our Service may contain links to third-party websites or services that are not owned or controlled by ScreenToVideo.",
                                     "We have no control over, and assume no responsibility for, the content, privacy policies, or practices of any third-party websites or services. Therefore, you are fully responsible for any risks you encounter while browsing third-party websites.",
                                     "We strongly advise you to read the terms and conditions and privacy policies of any third-party websites or services that you visit."]),
        ("Download Software", ["ScreenToVideo makes certain software products available via the ScreenToVideo.com website. Although you are free to download any software presented there, you are not allowed to reproduce or redistribute it."]),
        ("Feedback", ["We always value hearing from our users, whether they have questions, expectations, or comments, and are always interested in learning about your suggestions that can help us make our software better. If you choose to submit comments, ideas, or feedback, you agree that we are free to use them without any restriction or compensation to you."]),
    ]
    intro = "<p>Welcome to ScreenToVideo.com. By accessing this website, you agree to comply with and be bound by the following terms and conditions. ScreenToVideo reserves the right to change these terms and conditions, and the products, services, prices, and programs mentioned on its site at any time, as well as seek all remedies in law and in equity for any violation of these terms and conditions.</p>"
    return legal("website-terms-and-conditions-of-use", "Terms of Use - ScreenToVideo", "Website Terms and Conditions of Use",
                 "The terms and conditions for using the ScreenToVideo website: copyright, indemnity, disclaimer, links to other websites, software downloads and feedback.", S, intro)


def notfound():
    # 404 is served from arbitrary paths, so it uses absolute URLs to the site root.
    p = Page("")
    p.r = "/"  # GitHub Pages / most hosts serve 404.html at any depth
    content = page_hero(p, "404", "This page <span class=\"grad\">doesn’t exist</span>", "The page you’re looking for may have moved. Try one of these instead.",
                        extra=f'<div class="btns" style="margin-top:28px"><a class="btn btn-primary" href="/">Go home</a><a class="btn btn-ghost" href="/features/">Features</a><a class="btn btn-ghost" href="/support/">Support</a></div>')
    return write("404", layout(p, title="Page not found - ScreenToVideo", desc="This page doesn’t exist. Find ScreenToVideo features, pricing, downloads and support from here.", content=content, page="404", robots="noindex"))


BUILD_DATE = __import__("datetime").date.today().isoformat()


def llms_txt():
    # Plain-language summary for AI assistants and AI search (llmstxt.org format)
    feats = lambda lst: "\n".join(f"- {t}: {d}" for _, t, d, *_ in lst)
    faq = "\n".join(f"### {q}\n{a}\n" for q, a in FAQ)
    help_links = "\n".join(f"- [{sec['title']}]({SITE}/help/{sec['file']})" for sec in help_toc())
    return f"""# ScreenToVideo

> ScreenToVideo is a screen recorder and video editor for Windows (11, 10, 8 and 7, 64-bit) with AI background blur, AI virtual backgrounds, AI silence removal and AI voice-overs (Google Cloud Text-to-Speech, using the customer's own Google Cloud account, or built-in Microsoft voices). It is a one-time purchase ($119 regular, $79 educational) with free lifetime upgrades and a free, fully featured 15-day trial.

ScreenToVideo is made by {COMPANY} and has been developed since 2019. It records the screen (full screen, a window or a region), webcam, microphone and system sound, and includes a full video editor for adding text, shapes, zoom and pan, effects, transitions, animations, captions and interactivity. Videos export to MP4, WMV, animated GIF, animated WebP, audio-only formats, an interactive HTML player, or directly to Google Drive. It is rated {RATING} out of 5 on AppSumo.

## Key facts
- Platform: Windows only (64-bit Windows 11, 10, 8 or 7). No Mac version.
- Price: one-time license, no subscription. Regular $119, Educational $79. Multi-license discounts from {discount_range(0)} (2-5) to {discount_range(3)} (21+). 30-day money-back guarantee.
- Free trial: 15 days, all features; exported videos are watermarked until a license is registered.
- Licensing: per user; the registered user can install it on all of their own computers.
- System requirements: Intel Core i-series or AMD Phenom II, 2 GB RAM, 500 MB disk space, 1280x720 screen.
- Current version: {VERSION}.

## Pages
- [Home]({SITE}/): overview, AI tools, reviews and pricing
- [Features]({SITE}/features/): all {len(F_RECORD) + len(F_EDIT) + len(F_PUBLISH) + len(F_EASY)} features with demo videos
- [Pricing]({SITE}/buy/): licenses and discounts
- [Download]({SITE}/download/): free 15-day trial
- [Video tutorials]({SITE}/how-to/)
- [Changelog]({SITE}/changelog/): every release since 2019
- [System requirements]({SITE}/system-requirements/)
- [Help center]({SITE}/help/): step-by-step guides
- [Contact]({SITE}/contact/): {SUPPORT}

## Help center
{help_links}

## Recording features
{feats(F_RECORD)}

## Editing features
{feats(F_EDIT)}

## Publishing features
{feats(F_PUBLISH)}

## Frequently asked questions
{faq}"""


def extras():
    (ROOT / "favicon.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><circle cx="16" cy="16" r="13.6" fill="none" stroke="#ff5a4e" stroke-width="3.2"/><circle cx="16" cy="16" r="8.2" fill="#ff5a4e"/></svg>\n')
    pages = ["", "features/", "how-to/", "support/", "buy/", "download/", "changelog/", "contact/", "system-requirements/",
             "troubleshooting/", "eula/", "privacy-policy/", "website-terms-and-conditions-of-use/"]
    sm = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + "".join(
        f"  <url><loc>{SITE}/{x}</loc><lastmod>{BUILD_DATE}</lastmod></url>\n" for x in pages + ["help/"] + ["help/" + pg["file"] for sec in help_toc() for pg in [sec] + sec["pages"]]) + "</urlset>\n"
    (ROOT / "sitemap.xml").write_text(sm)
    ai_bots = ["GPTBot", "OAI-SearchBot", "ChatGPT-User", "ClaudeBot", "Claude-SearchBot", "Claude-User", "PerplexityBot", "Google-Extended", "Applebot-Extended", "CCBot"]
    (ROOT / "robots.txt").write_text("User-agent: *\nAllow: /\n\n# AI search and assistants are welcome\n"
                                     + "".join(f"User-agent: {bot}\nAllow: /\n\n" for bot in ai_bots)
                                     + f"Sitemap: {SITE}/sitemap.xml\n")
    (ROOT / "llms.txt").write_text(llms_txt())



# --------------------------------------------------------------------------
# Help center: /help/*.htm (same URLs as the old HelpNDoc help, so links from the app keep working).
# Articles are clean HTML fragments in _src/help/ (first imported by _src/import_help.py), images in /help/images/.
# --------------------------------------------------------------------------
HELP_DIR = ROOT / "_src/help"
HELP_ICONS = {f: ic for ic, _, f in HELP_TOPICS}


def help_covers(f):
    # One line describing what a help section covers, from its sub-page titles
    sec = next((x for x in help_toc() if x["file"] == f), None)
    names = [pg["title"] for pg in (sec["pages"] if sec else [])]
    if not names:
        return "Step-by-step guide in our help center."
    names = [names[0]] + [n[0].lower() + n[1:] if not n[:2].isupper() else n for n in names[1:]]
    if len(names) > 5:
        return ", ".join(names[:5]) + " and more."
    return (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]) + "."


def help_toc():
    return [sec for sec in json.loads((HELP_DIR / "toc.json").read_text()) if sec["file"] != "overview.htm"]


def help_body(f):
    return (HELP_DIR / f.replace(".htm", ".html")).read_text()


def help_text(frag):
    t = re.sub(r"<summary>.*?</summary>", " ", frag)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def help_search(p, big=False):
    return (f'<div class="hc-search{" hc-search-lg" if big else ""}" data-index="search.json">'
            f'{icon(p, "zoom")}<input type="search" placeholder="Search the help center" aria-label="Search the help center" autocomplete="off">'
            f'<div class="hc-res" role="listbox" hidden></div></div>')


def help_nav(p, toc, cur):
    out = ""
    for sec in toc:
        here = cur == sec["file"] or any(pg["file"] == cur for pg in sec["pages"])
        cur_a = lambda f: ' aria-current="page"' if f == cur else ""
        sub = "".join(f'<li><a href="{pg["file"]}"{cur_a(pg["file"])}>{e(pg["title"])}</a></li>' for pg in sec["pages"])
        out += (f'<li class="hc-sec{" on" if here else ""}"><a href="{sec["file"]}"{cur_a(sec["file"])}>{icon(p, HELP_ICONS.get(sec["file"], "book"))}'
                f'<span>{e(sec["title"])}</span></a>{f"<ul>{sub}</ul>" if sub else ""}</li>')
    return (f'<nav class="hc-nav" aria-label="Help contents"><details class="hc-navd" open><summary>{icon(p, "menu")}Contents</summary>'
            f'<ul>{out}</ul></details></nav>')


def help_contact(p):
    return f"""<div class="hc-contact">
  <span class="ic-box c">{icon(p, 'chat')}</span>
  <div><h2>Still need help?</h2><p>Email us at <a href="mailto:{SUPPORT}">{SUPPORT}</a> and we’ll get back to you within 24 hours.</p></div>
  <a class="btn btn-ghost btn-sm" href="{p.u('contact/')}">Contact support{icon(p, 'arrow')}</a>
</div>"""


def help_home(p, toc, f):
    cards = ""
    for i, sec in enumerate(toc):
        links = "".join(f'<li><a href="{pg["file"]}">{e(pg["title"])}</a></li>' for pg in sec["pages"])
        cards += (f'<article class="card hc-card rv" style="--rd:{(i % 3) * .05}s"><a class="hc-card-h" href="{sec["file"]}">'
                  f'<span class="ic-box{" c" if i % 2 else ""}">{icon(p, HELP_ICONS.get(sec["file"], "book"))}</span><h2>{e(sec["title"])}</h2></a>'
                  f'<ul>{links}</ul></article>')
    content = f"""<section class="phero center hc-hero"><div class="glow"></div><div class="wrap">
  <div class="eyebrow rv">Help Center</div>
  <h1 class="rv" style="--rd:.05s">How can we <span class="grad">help?</span></h1>
  <p class="lead rv" style="--rd:.1s">Step-by-step guides for recording, editing and exporting with ScreenToVideo.</p>
  <div class="rv" style="--rd:.15s">{help_search(p, big=True)}</div>
</div></section>
<section class="sec-sm" style="padding-top:0"><div class="wrap">
  <div class="grid g3 hc-cards">{cards}</div>
  <div class="grid g3" style="margin-top:16px">
    <a class="card card-spot rv" href="{p.u('how-to/')}"><span class="ic-box">{icon(p, 'play')}</span><h3>Video tutorials</h3><p>Watch short tutorials on recording, editing, adding text-to-speech, and exporting your project.</p></a>
    <a class="card card-spot rv" style="--rd:.05s" href="{p.u('troubleshooting/')}"><span class="ic-box c">{icon(p, 'wrench')}</span><h3>Troubleshooting</h3><p>Install the Windows components required for recording and playback.</p></a>
    <a class="card card-spot rv" style="--rd:.1s" href="mailto:{SUPPORT}"><span class="ic-box">{icon(p, 'mail')}</span><h3>Email support</h3><p>Can’t find an answer? Write to <span class="link">{SUPPORT}</span> and we’ll reply within 24 hours.</p></a>
  </div>
</div></section>"""
    out = ROOT / "help" / f
    out.write_text(layout(p, title="Help Center - ScreenToVideo", canon=f"{SITE}/help/", crumbs=[("Help Center", SITE + "/help/")],
                          desc="ScreenToVideo help center: step-by-step guides for recording your screen, editing your project and exporting your video.",
                          content=content, active="support/", page="help", track="help"), encoding="utf-8")
    return out


def help_page(p, toc, flat, i):
    pg, sec = flat[i]
    f = pg["file"]
    body = help_body(f)
    crumbs = f'<a href="./">Help Center</a>' + (f'{icon(p, "arrow")}<a href="{sec["file"]}">{e(sec["title"])}</a>' if sec else "")
    extra = ""
    if pg.get("pages"):
        extra = '<div class="hc-kids">' + "".join(
            f'<a class="hc-kid" href="{c["file"]}"><span>{e(c["title"])}</span>{icon(p, "arrow")}</a>' for c in pg["pages"]) + "</div>"
    elif sec:
        sib = [c for c in sec["pages"] if c["file"] != f]
        if sib:
            extra = '<div class="hc-also"><h2>See also</h2><ul>' + "".join(f'<li><a href="{c["file"]}">{e(c["title"])}</a></li>' for c in sib) + "</ul></div>"
    prev = flat[i - 1][0] if i > 0 else None
    nxt = flat[i + 1][0] if i + 1 < len(flat) else None
    pn = '<nav class="hc-pn" aria-label="Previous and next article">'
    pn += (f'<a class="hc-prev" href="{prev["file"]}"><small>Previous</small><span>{icon(p, "arrow", "i back")}{e(prev["title"])}</span></a>' if prev else "<span></span>")
    pn += (f'<a class="hc-next" href="{nxt["file"]}"><small>Next</small><span>{e(nxt["title"])}{icon(p, "arrow")}</span></a>' if nxt else "<span></span>")
    pn += "</nav>"
    txt = help_text(body)
    if txt:
        desc = txt[:155].rsplit(" ", 1)[0] + "…" if len(txt) > 155 else txt
    else:
        # screenshot-only article: describe it and give the images meaningful alt text
        desc = f'{pg["title"]} in ScreenToVideo, explained step by step with annotated screenshots.'
        body = body.replace('<img alt ', f'<img alt="{e(pg["title"])} in ScreenToVideo" ')
    content = f"""<section class="hc"><div class="wrap hc-grid">
  <aside class="hc-side">{help_search(p)}{help_nav(p, toc, f)}</aside>
  <article class="hc-main">
    <nav class="hc-crumbs" aria-label="Breadcrumb">{crumbs}</nav>
    <h1>{e(pg["title"])}</h1>
    <div class="hc-body">{body}</div>
    {extra}
    {pn}
    {help_contact(p)}
  </article>
</div></section>"""
    out = ROOT / "help" / f
    trail = [("Help Center", SITE + "/help/")] + ([(sec["title"], f"{SITE}/help/{sec['file']}")] if sec else []) + [(pg["title"], f"{SITE}/help/{f}")]
    art = ld_json({"@context": "https://schema.org", "@type": "TechArticle", "headline": pg["title"], "description": desc,
                   "url": f"{SITE}/help/{f}", "inLanguage": "en", "about": {"@type": "SoftwareApplication", "name": "ScreenToVideo"},
                   "author": {"@type": "Organization", "name": "ScreenToVideo", "url": SITE + "/"},
                   "publisher": {"@type": "Organization", "name": "ScreenToVideo", "url": SITE + "/"}})
    out.write_text(layout(p, title=f'{pg["title"]} - ScreenToVideo Help', canon=f"{SITE}/help/{f}", desc=desc, crumbs=trail, ld=art,
                          content=content, active="support/", page="help", track="help"), encoding="utf-8")
    return out


def help_center():
    p = Page("help")
    toc = help_toc()
    flat = []
    for sec in toc:
        flat.append((sec, None))
        flat += [(pg, sec) for pg in sec["pages"]]
    built = [help_home(p, toc, "index.html"), help_home(p, toc, "overview.htm")]
    built += [help_page(p, toc, flat, i) for i in range(len(flat))]
    idx = [{"t": pg["title"], "f": pg["file"], "s": sec["title"] if sec else "", "x": help_text(help_body(pg["file"]))} for pg, sec in flat]
    (ROOT / "help/search.json").write_text(json.dumps(idx, ensure_ascii=False, separators=(",", ":")))
    return built


if __name__ == "__main__":
    built = [home(), features(), howto(), install(), support(), buy(), download(), changelog(), sysreq(), troubleshooting(),
             contact(), uninstall(), gdrive(), eula(), privacy(), terms(), notfound()]
    built += [redeem(*r) for r in REDEEM]
    built += help_center()
    extras()
    for b in built:
        print(f"{b.relative_to(ROOT)}  {b.stat().st_size / 1024:.1f} KB")
