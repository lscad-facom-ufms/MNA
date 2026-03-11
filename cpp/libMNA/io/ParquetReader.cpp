#include <filesystem>
#include <memory>
#include <vector>

#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/ipc/reader.h>

#include <arrow/array.h>
#include <arrow/table.h>
#include <arrow/record_batch.h>
import mna.io.parquet_reader;

namespace mna {

namespace fs = std::filesystem;

arrow::Result<int64_t>
ParquetReader::get_num_rows(fs::path file_path){

  std::shared_ptr<arrow::io::ReadableFile> input_file;
  ARROW_ASSIGN_OR_RAISE(input_file, arrow::io::ReadableFile::Open(file_path.string()));
  std::unique_ptr<parquet::ParquetFileReader> reader = parquet::ParquetFileReader::Open(input_file);

  std::shared_ptr<parquet::FileMetaData> metadata = reader->metadata();
  return metadata->num_rows();
  
}
arrow::Result<std::shared_ptr<arrow::Table>>
ParquetReader::read_file(fs::path file_path){

  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(file_path.string()));

  // Open Parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  ARROW_ASSIGN_OR_RAISE(arrow_reader, parquet::arrow::OpenFile(input, pool));

  // Read entire file as a single Arrow table
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(&table));

  return table;
}

void
ParquetReader::read_vertexes(fs::path nodes_file, VertexCallback consume_vertex){

  auto table = read_file(nodes_file);

  arrow::TableBatchReader batch_reader(*table);
  std::shared_ptr<arrow::RecordBatch> batch;

  while (true) {
    arrow::Status status = batch_reader.ReadNext(&batch);
    if (!status.ok()) { break; }
    if (batch == nullptr) { break; }

    auto col_resources = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("R"));
    auto col_bandwidth = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("B"));
    auto col_busy = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("Busy"));
    auto col_inactive = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("Inactive"));

    for(int64_t i = 0; i < batch->num_rows(); ++i){
      int64_t R = col_resources->Value(i);
      int64_t B = col_bandwidth->Value(i);
      int64_t busy = col_busy->Value(i);
      int64_t inactive = col_inactive->Value(i);

      consume_vertex(R, B, busy, inactive);
    }
  }
}
void
ParquetReader::read_edges(std::vector<fs::path> edges_files, EdgeCallback consume_edge){

  for (auto& file : edges_files){

    auto table = read_file(file);

    arrow::TableBatchReader batch_reader(*table);
    std::shared_ptr<arrow::RecordBatch> batch;

    while (true) {
      arrow::Status status = batch_reader.ReadNext(&batch);
      if (!status.ok()) { break; }
      if (batch == nullptr) { break; }

      auto col_source = std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName("source"));
      auto col_target = std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName("target"));
      auto col_latency = std::static_pointer_cast<arrow::Int32Array>(batch->GetColumnByName("latency"));

      for(int64_t i = 0; i < batch->num_rows(); ++i){
        int64_t source = col_source->Value(i);
        int64_t target = col_target->Value(i);
        int64_t latency = col_latency->Value(i);

        consume_edge(source, target, latency);
      }
    }
  }
}
void
ParquetReader::read_jobs(fs::path jobs_file, JobCallback consume_job){

  auto table = read_file(jobs_file);

  arrow::TableBatchReader batch_reader(*table);
  std::shared_ptr<arrow::RecordBatch> batch;

  while (true) {
    arrow::Status status = batch_reader.ReadNext(&batch);
    if (!status.ok()) { break; }
    if (batch == nullptr) { break; }

    auto col_resources = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("jr"));
    auto col_bandwidth = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("jb"));
    auto col_latency = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("jl"));
    auto col_origin = std::static_pointer_cast<arrow::Int64Array>(batch->GetColumnByName("jo"));

    for(int64_t i = 0; i < batch->num_rows(); ++i){
      int64_t jr = col_resources->Value(i);
      int64_t jb = col_bandwidth->Value(i);
      int64_t jl = col_latency->Value(i);
      int64_t jo = col_origin->Value(i);

      consume_job(jr, jb, jl, jo);
    }
  }
}
}
