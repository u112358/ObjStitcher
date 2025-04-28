from manim import *


class ObjStitcherDemo(Scene):
    def construct(self):
        # Title
        title = Text("ObjStitcher Algorithm Demo", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # ---------------------------
        # Step 1: Accumulating Frames in the Buffer
        # ---------------------------
        # Create several "frame" rectangles that simulate incoming camera frames.
        frame_width = 2
        frame_height = 1
        num_frames = 6  # Total frames to simulate

        # Create a VGroup to hold the frames.
        frames = VGroup()
        for i in range(num_frames):
            # Create a rectangle for each frame.
            frame_rect = Rectangle(width=frame_width, height=frame_height,
                                   stroke_color=BLUE, fill_color=BLUE, fill_opacity=0.5)
            # Label the frame
            label = Text(
                f"Frame {i+1}", font_size=24).move_to(frame_rect.get_center())
            frame = VGroup(frame_rect, label)
            frames.add(frame)

        # Arrange frames vertically to represent the buffer's accumulation.
        frames.arrange(DOWN, buff=0.2).to_edge(LEFT)

        # Draw an outline box around the group to visually represent the buffer.
        buffer_box = SurroundingRectangle(frames, color=YELLOW, buff=0.3)
        buffer_label = Text("Buffer", font_size=24).next_to(buffer_box, UP)

        self.play(LaggedStartMap(FadeIn, frames, shift=UP, lag_ratio=0.2))
        self.play(Create(buffer_box), Write(buffer_label))
        self.wait(1)

        # ---------------------------
        # Step 2: Seeking for a Mark ("seek_mark" mode)
        # ---------------------------
        # Display a mode text. This represents the algorithm being in "seek_mark" mode.
        seek_text = Text("Mode: seek_mark", font_size=28,
                         color=GREEN).to_edge(RIGHT)
        self.play(Write(seek_text))
        self.wait(1)

        # Highlight one of the frames where a "mark" is detected.
        # (For the sake of demonstration, we choose the third frame.)
        detected_frame = frames[2]
        highlight = SurroundingRectangle(detected_frame, color=RED, buff=0.1)
        # Draw a horizontal red line inside the frame to represent the detected mark.
        mark_line = Line(start=detected_frame.get_left(
        ), end=detected_frame.get_right(), color=RED).shift(0.2 * UP)
        mark_label = Text("Mark Detected", font_size=20,
                          color=RED).next_to(highlight, UP)
        self.play(Create(highlight), Create(mark_line), Write(mark_label))
        self.wait(1)

        # ---------------------------
        # Step 3: Transition to "check_mark" Mode
        # ---------------------------
        # Replace mode text to indicate mode change.
        check_text = Text("Mode: check_mark", font_size=28,
                          color=ORANGE).to_edge(RIGHT)
        self.play(ReplacementTransform(seek_text, check_text))
        self.wait(1)

        # ---------------------------
        # Step 4: Extracting the Object Using a Sliding Window
        # ---------------------------
        # Draw a rectangle that overlays a subset of the buffer to simulate the sliding window extraction.
        # Here we pick a group of frames (e.g., frames 2,3,4) to form the output object.
        extraction_group = VGroup(frames[1], frames[2], frames[3])
        extraction_box = SurroundingRectangle(
            extraction_group, color=WHITE, buff=0.15)
        extraction_label = Text("Extracted Object", font_size=20, color=WHITE).move_to(
            extraction_box.get_center())
        self.play(Create(extraction_box), Write(extraction_label))
        self.wait(1)

        # ---------------------------
        # Step 5: Update the Buffer and Show Final Stitched Object
        # ---------------------------
        # Fade out the frames that have been processed (simulate removing them from the buffer).
        used_frames = VGroup(frames[0], frames[1], frames[2], frames[3])
        self.play(FadeOut(used_frames))
        self.wait(1)

        # Create a copy of the extraction box to represent the final output.
        final_obj = extraction_box.copy().shift(RIGHT * 3)
        final_obj.set_color(GREEN)
        final_label = Text("Final Stitched Object", font_size=24,
                           color=GREEN).next_to(final_obj, UP)
        self.play(FadeIn(final_obj), Write(final_label))
        self.wait(2)

        # End screen / Conclusion
        conclusion = Text("Demo Complete", font_size=32).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(2)
