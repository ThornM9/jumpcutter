# jumpcutter
Automatically edits videos. Explanation here: https://www.youtube.com/watch?v=DQ8orIurGxw

Go here for a more polished version of this software that my friends and I have been working on fr the last year or so: https://jumpcutter.com/

Since my GitHub is more like a dumping ground or personal journal, I'm not going to be actively updating this GitHub repo. But if you do want a version of jumpcutter that is actively being worked on, please do check on the version at https://jumpcutter.com/! There's way more developers fixing bugs and adding new features to that tool, and there's a developer's Discord server to discuss anything JC-related, so go check it out!

# Binaries
There are some pre-built binaries, using these are the easiest way to use jumpcutter. They don't require you to install any dependencies at all, including ffmpeg.

1. Download the binary for your operating system from the releases tab of the github repo
2. Open your terminal and navigate to the folder containing the binary executable.
3. Execute the binary using the usage instructions below or in the explainer video.

Basic usage:

`jumpcutter --input_file <input media path> --sounded_speed <speed multiplier> --silent_speed <speed multiplier>`

Process a folder of media:

`jumpcutter --input_folder <input directory> --sounded_speed <speed multiplier> --silent_speed <speed multiplier>`

More information and configuration options:

`jumpcutter --help`

If your OS doesn't have a pre-built binary available and you'd like to add one for others to use, you can follow the build instructions below.

1. Install pyinstaller with `pip install pyinstaller`
2. Download the ffmpeg binary for your OS to the jumpcutter repo folder
3. Run `pyinstaller -F --add-data "ffmpeg:ffmpeg" jumpcutter.py` (this command might change, check pysintaller docs for more info)
4. Create a pull request with a link to the binary and I'll add it to the releases.

## Some heads-up:

It uses Python 3.

It works on Ubuntu and Windows 10. (It might work on other OSs too, we just haven't tested it yet.)

This program relies heavily on ffmpeg. It will start subprocesses that call ffmpeg, so be aware of that! This only applies when running the source Python program.

As the program runs, it saves every frame of the video as an image file in a
temporary folder. If your video is long and has high resolution and frame rate, this could take a LOT of space.

## Building with nix
`nix-build` to get a script with all the libraries and ffmpeg, `nix-build -A bundle` to get a single binary.

## Results
Personally, I use this tool to watch lectures when I'm really far behind on my course. To show how much time you save using this tool, I ran it on an entire semester of lecture videos for one math ssubject.

Sounded speed: 1.5x
Silent speed: 8x
Total duration of content: 1329 minutes
Total duration without jumpcutter at 1.5x: 886 minutes
Total duration with jumpcutter: 632 minutes

That's a saving of over 4.2 hours over 22.2 hours of original content, 29% faster than watching at flat 1.5x speed.
