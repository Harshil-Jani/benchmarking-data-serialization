# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Sample

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Person(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Person()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsPerson(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Person
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Person
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Person
    def Id(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    # Person
    def Email(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Person
    def Samples(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # Person
    def SamplesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float64Flags, o)
        return 0

    # Person
    def SamplesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Person
    def SamplesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

def PersonStart(builder):
    builder.StartObject(4)

def Start(builder):
    PersonStart(builder)

def PersonAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    PersonAddName(builder, name)

def PersonAddId(builder, id):
    builder.PrependInt64Slot(1, id, 0)

def AddId(builder, id):
    PersonAddId(builder, id)

def PersonAddEmail(builder, email):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(email), 0)

def AddEmail(builder, email):
    PersonAddEmail(builder, email)

def PersonAddSamples(builder, samples):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(samples), 0)

def AddSamples(builder, samples):
    PersonAddSamples(builder, samples)

def PersonStartSamplesVector(builder, numElems):
    return builder.StartVector(8, numElems, 8)

def StartSamplesVector(builder, numElems):
    return PersonStartSamplesVector(builder, numElems)

def PersonEnd(builder):
    return builder.EndObject()

def End(builder):
    return PersonEnd(builder)
