//============================================================================
// Activation Buffer - Store Conv1 Outputs for XAI Analysis
// Stores 12 positions × 8 filters = 96 activation values (INT8)
// Used for feature importance tracking in explainable AI
//============================================================================

module activation_buffer #(
    parameter NUM_FILTERS = 8,                // Number of Conv1 filters
    parameter SEQ_LEN     = 12,               // Sequence length after Conv1
    parameter DATA_WIDTH  = 8                 // Activation data width (INT8)
)(
    input  wire                 clk,
    input  wire                 rst,
    
    // Write interface (from Conv1)
    input  wire                 write_en,
    input  wire [3:0]           write_filter,  // Filter address (0-7)
    input  wire [3:0]           write_seq,     // Sequence position (0-11)
    input  wire signed [DATA_WIDTH-1:0] write_data,
    
    // Read interface (for XAI scanner)
    input  wire                 read_en,
    input  wire [3:0]           read_filter,   // Filter address (0-7)
    input  wire [3:0]           read_seq,      // Sequence position (0-11)
    output reg  signed [DATA_WIDTH-1:0] read_data,
    
    // Status
    output reg                  buffer_ready   // Buffer initialized and ready
);

    // Memory array: 96 locations (12 × 8)
    // Packed layout: addr = seq * NUM_FILTERS + filter
    localparam BUFFER_DEPTH = SEQ_LEN * NUM_FILTERS;  // 96
    localparam ADDR_WIDTH   = 7;  // ceil(log2(96)) = 7
    
    reg signed [DATA_WIDTH-1:0] activation_map [0:BUFFER_DEPTH-1];
    
    wire [ADDR_WIDTH-1:0] write_addr;
    wire [ADDR_WIDTH-1:0] read_addr;
    
    // Address calculation
    assign write_addr = (write_seq * NUM_FILTERS) + write_filter;
    assign read_addr  = (read_seq * NUM_FILTERS) + read_filter;
    
    //========================================================================
    // Sequential Write/Read Operations
    //========================================================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            buffer_ready <= 1'b0;
            read_data    <= {DATA_WIDTH{1'b0}};
        end
        else begin
            buffer_ready <= 1'b1;
            
            // Write operation
            if (write_en) begin
                activation_map[write_addr] <= write_data;
            end
            
            // Read operation (combinational output registered)
            if (read_en) begin
                read_data <= activation_map[read_addr];
            end
        end
    end
    
    //========================================================================
    // Synthesis Notes
    //========================================================================
    // For FPGA synthesis, this will infer:
    // - Distributed RAM (LUT-based) for small buffers (~100 LUTs)
    // - Or BRAM if depth exceeds threshold
    // 
    // Resource estimate:
    // - 96 × 8 bits = 768 bits storage
    // - ~100 LUTs for distributed RAM implementation
    // - Read/write in same cycle supported (different addresses)
    
endmodule
