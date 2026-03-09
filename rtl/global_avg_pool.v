//============================================================================
// Global Average Pooling - Reduces spatial dimensions to 1 per channel
// Accumulates values per channel and divides by feature length
// Uses shift-based division for power-of-2 lengths
//============================================================================

module global_avg_pool #(
    parameter DATA_WIDTH    = 8,              // Input data width
    parameter ACC_WIDTH     = 32,             // Accumulator width
    parameter INPUT_LENGTH  = 63,             // Input length per channel (after maxpool)
    parameter NUM_FILTERS   = 8               // Number of channels/filters
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,          // Start GAP
    output reg                  done,           // GAP complete
    
    // Streaming input
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                 valid_in,
    
    // Streaming output (one value per channel)
    output reg signed [DATA_WIDTH-1:0] data_out,
    output reg                  valid_out
);

    // Accumulator per channel
    reg signed [ACC_WIDTH-1:0] channel_acc [0:NUM_FILTERS-1];
    
    // Counters
    reg [2:0] channel_cnt;      // Current channel (0-7)
    reg [7:0] pos_cnt;          // Position within channel
    reg [10:0] total_cnt;       // Total samples processed
    reg running;
    reg accumulating;
    reg dividing;
    
    // Division result
    reg signed [ACC_WIDTH-1:0] div_result;
    
    // Total samples: INPUT_LENGTH * NUM_FILTERS = 63 * 8 = 504
    localparam TOTAL_SAMPLES = INPUT_LENGTH * NUM_FILTERS;
    
    // Initialize channel accumulators
    integer i;
    initial begin
        for (i = 0; i < NUM_FILTERS; i = i + 1) begin
            channel_acc[i] = {ACC_WIDTH{1'b0}};
        end
    end

    //========================================================================
    // Global Average Pooling Logic
    // For division by INPUT_LENGTH (63), we use approximation:
    // 63 ≈ 64 = 2^6, so divide by right-shifting 6 bits
    // For exact division, a small lookup or iterative subtract could be used
    //========================================================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done        <= 1'b0;
            valid_out   <= 1'b0;
            data_out    <= {DATA_WIDTH{1'b0}};
            channel_cnt <= 3'd0;
            pos_cnt     <= 8'd0;
            total_cnt   <= 11'd0;
            running     <= 1'b0;
            accumulating<= 1'b0;
            dividing    <= 1'b0;
            
            // Reset accumulators
            channel_acc[0] <= {ACC_WIDTH{1'b0}};
            channel_acc[1] <= {ACC_WIDTH{1'b0}};
            channel_acc[2] <= {ACC_WIDTH{1'b0}};
            channel_acc[3] <= {ACC_WIDTH{1'b0}};
            channel_acc[4] <= {ACC_WIDTH{1'b0}};
            channel_acc[5] <= {ACC_WIDTH{1'b0}};
            channel_acc[6] <= {ACC_WIDTH{1'b0}};
            channel_acc[7] <= {ACC_WIDTH{1'b0}};
        end
        else begin
            done      <= 1'b0;
            valid_out <= 1'b0;
            
            if (start && !running) begin
                running      <= 1'b1;
                accumulating <= 1'b1;
                dividing     <= 1'b0;
                channel_cnt  <= 3'd0;
                pos_cnt      <= 8'd0;
                total_cnt    <= 11'd0;
                
                // Reset accumulators
                channel_acc[0] <= {ACC_WIDTH{1'b0}};
                channel_acc[1] <= {ACC_WIDTH{1'b0}};
                channel_acc[2] <= {ACC_WIDTH{1'b0}};
                channel_acc[3] <= {ACC_WIDTH{1'b0}};
                channel_acc[4] <= {ACC_WIDTH{1'b0}};
                channel_acc[5] <= {ACC_WIDTH{1'b0}};
                channel_acc[6] <= {ACC_WIDTH{1'b0}};
                channel_acc[7] <= {ACC_WIDTH{1'b0}};
            end
            else if (accumulating) begin
                if (valid_in) begin
                    // Accumulate into current channel
                    channel_acc[channel_cnt] <= channel_acc[channel_cnt] + 
                                                {{(ACC_WIDTH-DATA_WIDTH){data_in[DATA_WIDTH-1]}}, data_in};
                    pos_cnt <= pos_cnt + 8'd1;
                    total_cnt <= total_cnt + 11'd1;
                    
                    // Check if completed one channel
                    if (pos_cnt == INPUT_LENGTH - 1) begin
                        // Move to next channel or finish
                        if (channel_cnt == NUM_FILTERS - 1) begin
                            // All channels accumulated, start division phase
                            accumulating <= 1'b0;
                            dividing     <= 1'b1;
                            channel_cnt  <= 3'd0;
                        end
                        else begin
                            channel_cnt <= channel_cnt + 3'd1;
                            pos_cnt     <= 8'd0;
                        end
                    end
                end
            end
            else if (dividing) begin
                // Perform division (average) for current channel
                // Using shift: divide by 64 (2^6) as approximation for 63
                // For exact: could use (acc * 1024) / 63 then shift, but shift is simpler
                div_result = channel_acc[channel_cnt] >>> 6;  // Divide by 64
                
                // Saturate to INT8
                if (div_result > 127)
                    data_out <= 8'd127;
                else if (div_result < -128)
                    data_out <= 8'd128;
                else
                    data_out <= div_result[DATA_WIDTH-1:0];
                    
                valid_out <= 1'b1;
                
                // Move to next channel
                if (channel_cnt == NUM_FILTERS - 1) begin
                    // All channels output
                    running  <= 1'b0;
                    dividing <= 1'b0;
                    done     <= 1'b1;
                end
                else begin
                    channel_cnt <= channel_cnt + 3'd1;
                end
            end
        end
    end

endmodule
