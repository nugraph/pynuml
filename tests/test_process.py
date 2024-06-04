"""Test pynuml graph processing"""
import pynuml

def test_process_uboone():
    """Test graph processing with MicroBooNE open data release"""
    f = pynuml.io.File("/raid/nugraph/uboone-opendata/uboone-opendata.evt.h5")
    processor = pynuml.process.HitGraphProducer(
        file=f,
        semantic_labeller=pynuml.labels.StandardLabels(),
        event_labeller=pynuml.labels.FlavorLabels(),
        label_vertex=True)
    f.read_data(0, 100)
    evts = f.build_evt()
    for evt in evts:
        processor(evt)
