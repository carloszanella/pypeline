from trends_ni.entities import SubjectFMRI


def test_subject_fmri():
    subj = SubjectFMRI(id=10001)
    subj.load_data()
    assert subj.fmri_map.compute().any()
