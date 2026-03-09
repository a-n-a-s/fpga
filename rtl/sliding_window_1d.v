//============================================================================
// Sliding Window 1D - 3-sample window generator for Conv1D
// Generates overlapping windows from streaming input
// Kernel size = 3, stride = 1, padding = valid
//============================================================================

module sliding_window_1d #(
    parameter DATA_WIDTH    = 8,              // Input data width
    parameter INPUT_LENGTH  = 128,            // Input sequence length
    parameter KERNEL_SIZE   = 3,              // Convolution kernel size
    parameter NUM_FILTERS   = 8               // Number of filters (for timing)
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,          // Start window generation
    output reg                  done,           // Window generation complete
    
    // Streaming input
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                 valid_in,
    
    // Window outputs (3 samples for kernel size 3)
    output reg signed [DATA_WIDTH-1:0] win_sample0,  // Oldest sample
    output reg signed [DATA_WIDTH-1:0] win_sample1,  // Middle sample
    output reg signed [DATA_WIDTH-1:0] win_sample2,  // Newest sample
    output reg                  window_valid          // Window data valid
);

    // Shift register for sliding window
    reg signed [DATA_WIDTH-1:0] shift_reg0;
    reg signed [DATA_WIDTH-1:0] shift_reg1;
    reg signed [DATA_WIDTH-1:0] shift_reg2;
    
    // Counters
    reg [7:0] sample_cnt;
    reg [7:0] output_cnt;
    
    // Control
    reg running;
    reg init_done;
    
    // Output length for valid padding: INPUT_LENGTH - KERNEL_SIZE + 1
    // For 128 input, kernel 3: 128 - 3 + 1 = 126 outputs
    localparam OUTPUT_LENGTH = INPUT_LENGTH - KERNEL_SIZE + 1;

    //========================================================================
    // Sliding window logic
    //========================================================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            shift_reg0    <= {DATA_WIDTH{1'b0}};
            shift_reg1    <= {DATA_WIDTH{1'b0}};
            shift_reg2    <= {DATA_WIDTH{1'b0}};
            win_sample0   <= {DATA_WIDTH{1'b0}};
            win_sample1   <= {DATA_WIDTH{1'b0}};
            win_sample2   <= {DATA_WIDTH{1'b0}};
            sample_cnt    <= 8'd0;
            output_cnt    <= 8'd0;
            running       <= 1'b0;
            init_done     <= 1'b0;
            done          <= 1'b0;
            window_valid  <= 1'b0;
        end
        else begin
            done          <= 1'b0;  // Default clear done
            window_valid  <= 1'b0;  // Default clear valid
            
            if (start && !running) begin
                // Initialize operation
                running     <= 1'b1;
                init_done   <= 1'b0;
                sample_cnt  <= 8'd0;
                output_cnt  <= 8'd0;
                shift_reg0  <= {DATA_WIDTH{1'b0}};
                shift_reg1  <= {DATA_WIDTH{1'b0}};
                shift_reg2  <= {DATA_WIDTH{1'b0}};
            end
            else if (running) begin
                if (valid_in) begin
                    // Shift register update
                    shift_reg0 <= shift_reg1;
                    shift_reg1 <= shift_reg2;
                    shift_reg2 <= data_in;
                    sample_cnt <= sample_cnt + 8'd1;
                    
                    // After filling kernel_size samples, output valid windows
                    if (sample_cnt >= KERNEL_SIZE - 1) begin
                        win_sample0  <= shift_reg0;
                        win_sample1  <= shift_reg1;
                        win_sample2  <= shift_reg2;
                        window_valid <= 1'b1;
                        output_cnt   <= output_cnt + 8'd1;
                        
                        // Check completion
                        if (output_cnt == OUTPUT_LENGTH - 1) begin
                            running  <= 1'b0;
                            done     <= 1'b1;
                        end
                    end
                end
            end
            else begin
                // Hold outputs
                window_valid <= 1'b0;
            end
        end
    end

endmodule
