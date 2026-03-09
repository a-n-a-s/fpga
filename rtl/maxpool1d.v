//============================================================================
// MaxPool1D - 1D Max Pooling
// Pool size = 2, stride = 2
// Sequential comparison for streaming input
//============================================================================

module maxpool1d #(
    parameter DATA_WIDTH    = 8,              // Input/Output data width
    parameter INPUT_LENGTH  = 126,            // Input length (after conv1)
    parameter POOL_SIZE     = 2,              // Pooling window size
    parameter STRIDE        = 2,              // Stride
    parameter NUM_FILTERS   = 8               // Number of filters/channels
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,          // Start pooling
    output reg                  done,           // Pooling complete
    
    // Streaming input
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                 valid_in,
    
    // Streaming output
    output reg signed [DATA_WIDTH-1:0] data_out,
    output reg                  valid_out
);

    // Output length: (INPUT_LENGTH - POOL_SIZE) / STRIDE + 1
    // For 126 input, pool 2, stride 2: (126-2)/2 + 1 = 63
    localparam OUTPUT_LENGTH = (INPUT_LENGTH - POOL_SIZE) / STRIDE + 1;
    localparam TOTAL_INPUTS  = INPUT_LENGTH * NUM_FILTERS;  // 126 * 8 = 1008
    localparam TOTAL_OUTPUTS = OUTPUT_LENGTH * NUM_FILTERS; // 63 * 8 = 504
    
    // Registers
    reg signed [DATA_WIDTH-1:0] max_val;
    reg signed [DATA_WIDTH-1:0] current_in;
    reg [7:0] pool_cnt;           // Count within pool window
    reg [10:0] in_sample_cnt;     // Total input samples
    reg [10:0] out_sample_cnt;    // Total output samples
    reg [2:0] filter_cnt;         // Current filter
    reg running;
    reg first_in_pool;
    
    //========================================================================
    // Sequential Max Pooling
    //========================================================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done          <= 1'b0;
            valid_out     <= 1'b0;
            data_out      <= {DATA_WIDTH{1'b0}};
            max_val       <= {DATA_WIDTH{1'b0}};
            current_in    <= {DATA_WIDTH{1'b0}};
            pool_cnt      <= 8'd0;
            in_sample_cnt <= 11'd0;
            out_sample_cnt<= 11'd0;
            filter_cnt    <= 3'd0;
            running       <= 1'b0;
            first_in_pool <= 1'b1;
        end
        else begin
            done      <= 1'b0;
            valid_out <= 1'b0;
            
            if (start && !running) begin
                running       <= 1'b1;
                in_sample_cnt <= 11'd0;
                out_sample_cnt<= 11'd0;
                filter_cnt    <= 3'd0;
                pool_cnt      <= 8'd0;
                first_in_pool <= 1'b1;
            end
            else if (running) begin
                if (valid_in) begin
                    current_in <= data_in;
                    
                    if (first_in_pool) begin
                        // First element in pool window - initialize max
                        max_val       <= data_in;
                        pool_cnt      <= 8'd1;
                        first_in_pool <= 1'b0;
                    end
                    else begin
                        // Compare with current max
                        if (data_in > max_val) begin
                            max_val <= data_in;
                        end
                        pool_cnt <= pool_cnt + 8'd1;
                        
                        // Check if pool window complete
                        if (pool_cnt == POOL_SIZE - 1) begin
                            // Output max value
                            data_out  <= max_val;
                            valid_out <= 1'b1;
                            out_sample_cnt <= out_sample_cnt + 11'd1;
                            
                            // Reset for next pool window
                            pool_cnt      <= 8'd0;
                            first_in_pool <= 1'b1;
                            
                            // Check completion
                            if (out_sample_cnt == TOTAL_OUTPUTS - 1) begin
                                running <= 1'b0;
                                done    <= 1'b1;
                            end
                        end
                    end
                    
                    in_sample_cnt <= in_sample_cnt + 11'd1;
                end
            end
            else begin
                valid_out <= 1'b0;
            end
        end
    end

endmodule
