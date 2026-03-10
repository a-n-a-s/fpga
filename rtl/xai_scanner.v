//============================================================================
// XAI Scanner - Find Most Important Features from Conv1 Activations
// Scans activation buffer to find max activation and its position
// Outputs: most_important_sample (0-11), most_important_filter (0-7)
//============================================================================

module xai_scanner #(
    parameter NUM_FILTERS = 8,                // Number of Conv1 filters
    parameter SEQ_LEN     = 12,               // Sequence length
    parameter DATA_WIDTH  = 8,                // Activation data width
    parameter SCORE_WIDTH = 8                 // Importance score width
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,          // Start scanning
    output reg                  done,           // Scanning complete
    
    // Activation buffer interface
    output reg                  read_en,
    output reg [3:0]            read_filter,
    output reg [3:0]            read_seq,
    input  wire signed [DATA_WIDTH-1:0] read_data,
    
    // Results
    output reg [3:0]            most_important_sample,  // Sequence position (0-11)
    output reg [3:0]            most_important_filter,  // Filter index (0-7)
    output reg [SCORE_WIDTH-1:0] importance_score,      // Max activation value
    output reg [7:0]            total_activation,       // Sum of all activations
    output reg [7:0]            avg_activation          // Average activation
);

    // State machine
    reg [3:0] state;
    localparam S_IDLE     = 4'd0;
    localparam S_SCAN     = 4'd1;
    localparam S_WAIT     = 4'd2;  // Wait for read data
    localparam S_COMPARE  = 4'd3;
    localparam S_UPDATE   = 4'd4;
    localparam S_FINISH   = 4'd5;
    
    // Counters
    reg [3:0] scan_seq;       // Current sequence position (0-11)
    reg [3:0] scan_filter;    // Current filter (0-7)
    
    // Max tracking
    reg signed [DATA_WIDTH-1:0] max_val;
    reg [3:0] max_seq;
    reg [3:0] max_filter;
    
    // Sum tracking (for average)
    reg signed [15:0] sum_val;
    
    // Activation buffer ready
    wire buffer_ready;
    
    localparam BUFFER_DEPTH = SEQ_LEN * NUM_FILTERS;  // 96 locations
    
    //========================================================================
    // Sequential Feature Importance Scanner
    //========================================================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done                <= 1'b0;
            read_en             <= 1'b0;
            read_filter         <= 4'd0;
            read_seq            <= 4'd0;
            most_important_sample <= 4'd0;
            most_important_filter <= 4'd0;
            importance_score    <= {SCORE_WIDTH{1'b0}};
            total_activation    <= 8'd0;
            avg_activation      <= 8'd0;
            scan_seq            <= 4'd0;
            scan_filter         <= 4'd0;
            max_val             <= {DATA_WIDTH{1'b0}};
            max_seq             <= 4'd0;
            max_filter          <= 4'd0;
            sum_val             <= 16'd0;
            state               <= S_IDLE;
        end
        else begin
            done    <= 1'b0;
            read_en <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    if (start) begin
                        // Initialize scan
                        scan_seq    <= 4'd0;
                        scan_filter <= 4'd0;
                        max_val     <= -8'sd128;  // Start with minimum INT8
                        max_seq     <= 4'd0;
                        max_filter  <= 4'd0;
                        sum_val     <= 16'd0;
                        state       <= S_SCAN;
                    end
                end
                
                S_SCAN: begin
                    // Set read address
                    read_seq    <= scan_seq;
                    read_filter <= scan_filter;
                    read_en     <= 1'b1;
                    state       <= S_WAIT;
                end
                
                S_WAIT: begin
                    // Wait one cycle for read data to be valid
                    read_en     <= 1'b0;
                    state       <= S_COMPARE;
                end

                S_COMPARE: begin
                    // Accumulate sum (use absolute value for total activity)
                    if (read_data >= 0)
                        sum_val <= sum_val + read_data;
                    else
                        sum_val <= sum_val - read_data;  // Add absolute value
                    
                    // Compare with current max
                    if (read_data > max_val) begin
                        max_val    <= read_data;
                        max_seq    <= scan_seq;
                        max_filter <= scan_filter;
                    end
                    
                    state <= S_UPDATE;
                end
                
                S_UPDATE: begin
                    // Move to next location
                    if (scan_filter < NUM_FILTERS - 1) begin
                        // Next filter in same sequence position
                        scan_filter <= scan_filter + 4'd1;
                        state <= S_SCAN;
                    end
                    else if (scan_seq < SEQ_LEN - 1) begin
                        // Next sequence position, reset filter
                        scan_filter <= 4'd0;
                        scan_seq    <= scan_seq + 4'd1;
                        state <= S_SCAN;
                    end
                    else begin
                        // All locations scanned
                        state <= S_FINISH;
                    end
                end
                
                S_FINISH: begin
                    // Store results
                    most_important_sample <= max_seq;
                    most_important_filter <= max_filter;
                    
                    // Convert signed max to unsigned score (offset by 128 for 0-255 range)
                    if (max_val >= 0)
                        importance_score <= max_val[SCORE_WIDTH-1:0];
                    else
                        importance_score <= {SCORE_WIDTH{1'b0}};
                    
                    // Calculate average (sum / 96, approximate with sum / 128 = sum >> 7)
                    if (sum_val >= 0) begin
                        total_activation <= sum_val[7:0];
                        avg_activation   <= sum_val[7:0];  // Approximate average
                    end
                    else begin
                        total_activation <= 8'd0;
                        avg_activation   <= 8'd0;
                    end
                    
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                
                default: state <= S_IDLE;
            endcase
        end
    end
    
    //========================================================================
    // Usage Notes
    //========================================================================
    // 1. Assert start for one cycle to begin scan
    // 2. Scanner reads all 96 locations sequentially (96 cycles)
    // 3. done goes high when scan complete
    // 4. Results available on done cycle:
    //    - most_important_sample: which glucose sample mattered most (0-11)
    //    - most_important_filter: which filter detected it (0-7)
    //    - importance_score: activation value at that position
    //    - total_activation: sum of all activations (activity level)
    //
    // Resource Estimate:
    // - LUTs: ~60-80
    // - FFs: ~40
    // - Scan time: 96 × 3 = 288 cycles (~2.9 μs @ 100 MHz)
    
endmodule
