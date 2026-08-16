from pipeline import AcousticGuardPipeline
from network_audio import NetworkAudioSource
from console_display import ConsoleDisplay
import config

display = ConsoleDisplay()
pipeline = AcousticGuardPipeline(on_event=display.handle_event, checkpoint_path="/home/acer01/Documents/keylogging/dataset/runs/20260816_074302/best_model.pt")
try:
    pipeline.run(NetworkAudioSource(host="0.0.0.0", port=config.NETWORK_PORT))
except KeyboardInterrupt:
    pass
finally:
    display.print_summary()


from console_display import GroundTruthTester

tester = GroundTruthTester(expected_text="helloworld")
pipeline = AcousticGuardPipeline(on_event=tester.handle_event, checkpoint_path="...")
# ...type "helloworld" on the capture device...
tester.print_report()
