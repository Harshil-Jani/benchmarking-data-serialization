import time
import json
import numpy as np
import statistics

import person_pb2
import flatbuffers
import MyData.Sample.Person
import MyData.Sample.AddressBook

def generate_sample_data(num_records):
    data = []
    for i in range(num_records):
        person = {
            "name": f"Person_{i}",
            "id": i,
            "email": f"person_{i}@example.com",
            "samples": list(np.random.rand(10))
        }
        data.append(person)
    return data

def benchmark_json(data):
    start_time = time.perf_counter()
    json_data = json.dumps({"people": data}).encode('utf-8')
    serialization_time = time.perf_counter() - start_time
    data_size = len(json_data)

    start_time = time.perf_counter()
    deserialized_data = json.loads(json_data)
    deserialization_time = time.perf_counter() - start_time
    assert deserialized_data['people'][0]['id'] == 0
    return serialization_time, deserialization_time, data_size

def benchmark_protobuf(data):
    address_book = person_pb2.AddressBook()
    for item in data:
        person = address_book.people.add()
        person.name = item["name"]
        person.id = item["id"]
        person.email = item["email"]
        person.samples.extend(item["samples"])

    start_time = time.perf_counter()
    proto_data = address_book.SerializeToString()
    serialization_time = time.perf_counter() - start_time
    data_size = len(proto_data)

    start_time = time.perf_counter()
    deserialized_proto = person_pb2.AddressBook()
    deserialized_proto.ParseFromString(proto_data)
    deserialization_time = time.perf_counter() - start_time
    assert deserialized_proto.people[0].id == 0
    return serialization_time, deserialization_time, data_size

def benchmark_flatbuffers(data):
    builder = flatbuffers.Builder(0)
    
    start_time = time.perf_counter()
    
    person_offsets = []
    for item in data:
        name = builder.CreateString(item["name"])
        email = builder.CreateString(item["email"])
        MyData.Sample.Person.PersonStartSamplesVector(builder, len(item["samples"]))
        for sample in reversed(item["samples"]):
            builder.PrependFloat64(sample)
        samples_vector = builder.EndVector()

        MyData.Sample.Person.PersonStart(builder)
        MyData.Sample.Person.PersonAddName(builder, name)
        MyData.Sample.Person.PersonAddId(builder, item["id"])
        MyData.Sample.Person.PersonAddEmail(builder, email)
        MyData.Sample.Person.PersonAddSamples(builder, samples_vector)
        person_offsets.append(MyData.Sample.Person.PersonEnd(builder))

    MyData.Sample.AddressBook.AddressBookStartPeopleVector(builder, len(person_offsets))
    for offset in reversed(person_offsets):
        builder.PrependUOffsetTRelative(offset)
    people_vector = builder.EndVector()

    MyData.Sample.AddressBook.AddressBookStart(builder)
    MyData.Sample.AddressBook.AddressBookAddPeople(builder, people_vector)
    address_book_offset = MyData.Sample.AddressBook.AddressBookEnd(builder)
    builder.Finish(address_book_offset)
    
    flatbuffer_data = builder.Output()
    serialization_time = time.perf_counter() - start_time
    data_size = len(flatbuffer_data)

    start_time = time.perf_counter()
    deserialized_root = MyData.Sample.AddressBook.AddressBook.GetRootAsAddressBook(flatbuffer_data, 0)
    deserialization_time = time.perf_counter() - start_time
    assert deserialized_root.People(0).Id() == 0
    return serialization_time, deserialization_time, data_size

def main():
    NUM_RECORDS = 10000
    NUM_RUNS = 100
    
    print(f"--- Running Benchmark for {NUM_RECORDS} records ({NUM_RUNS} runs) ---")
    
    sample_data = generate_sample_data(NUM_RECORDS)
    
    # Lists to store results from each run
    json_results = []
    proto_results = []
    flat_results = []

    for i in range(NUM_RUNS):
        print(f"Running iteration {i+1}/{NUM_RUNS}...", end='\r')
        json_results.append(benchmark_json(sample_data))
        proto_results.append(benchmark_protobuf(sample_data))
        flat_results.append(benchmark_flatbuffers(sample_data))
    print("\nBenchmark runs complete.\n")

    # Calculate averages
    avg_json_ser = statistics.mean([res[0] for res in json_results])
    avg_json_des = statistics.mean([res[1] for res in json_results])
    avg_json_size = statistics.mean([res[2] for res in json_results])

    avg_proto_ser = statistics.mean([res[0] for res in proto_results])
    avg_proto_des = statistics.mean([res[1] for res in proto_results])
    avg_proto_size = statistics.mean([res[2] for res in proto_results])

    avg_flat_ser = statistics.mean([res[0] for res in flat_results])
    avg_flat_des = statistics.mean([res[1] for res in flat_results])
    avg_flat_size = statistics.mean([res[2] for res in flat_results])

    print("--- Averaged Benchmark Results ---\n")
    print(f"{'Format':<15} | {'Avg Serialization (s)':<24} | {'Avg Deserialization (s)':<25} | {'Avg Data Size (bytes)':<20}")
    print("-" * 95)
    print(f"{'JSON':<15} | {avg_json_ser:<24.6f} | {avg_json_des:<25.6f} | {int(avg_json_size):<20,}")
    print(f"{'Protobuf':<15} | {avg_proto_ser:<24.6f} | {avg_proto_des:<25.6f} | {int(avg_proto_size):<20,}")
    print(f"{'FlatBuffers':<15} | {avg_flat_ser:<24.6f} | {avg_flat_des:<25.6f} | {int(avg_flat_size):<20,}")
    print("\n")


if __name__ == "__main__":
    main()
