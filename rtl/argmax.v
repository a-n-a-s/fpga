//============================================================================
// ArgMax - Compare 2 logits and output class with maximum value
// Simple comparator for 2-class classification
//============================================================================

module argmax #(
    parameter LOGIT_WIDTH   = 16,             // Input logit width (INT16)
    parameter NUM_CLASSES   = 2               // Number of output classes
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,          // Start argmax
    output reg                  done,           // Argmax complete
    
    // Input logits (sequential: logit0, then logit1)
    input  wire signed [LOGIT_WIDTH-1:0] logit_in,
    input  wire                 valid_in,
    
    // Output class
    output reg [1:0]          class_out,      // Predicted class (0 or 1)
    output reg                  valid_out
);

    // Registers for storing logits
    reg signed [LOGIT_WIDTH-1:0] logit0;
    reg signed [LOGIT_WIDTH-1:0] logit1;
    
    // State machine
    reg [1:0] state;
    localparam S_IDLE   = 2'd0;
    localparam S_LOAD0  = 2'd1;
    localparam S_LOAD1  = 2'd2;
    localparam S_CMP    = 2'd3;
    
    // Counter for sequential input
    reg [1:0] input_cnt;
    reg running;

    //========================================================================
    // ArgMax comparison logic
    //========================================================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done      <= 1'b0;
            valid_out <= 1'b0;
            class_out <= 2'd0;
            logit0    <= {LOGIT_WIDTH{1'b0}};
            logit1    <= {LOGIT_WIDTH{1'b0}};
            input_cnt <= 2'd0;
            running   <= 1'b0;
            state     <= S_IDLE;
        end
        else begin
            done      <= 1'b0;
            valid_out <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    if (start) begin
                        running   <= 1'b1;
                        input_cnt <= 2'd0;
                        state     <= S_LOAD0;
                    end
                end
                
                S_LOAD0: begin
                    if (valid_in) begin
                        logit0    <= logit_in;
                        input_cnt <= input_cnt + 2'd1;
                        state     <= S_LOAD1;
                    end
                end
                
                S_LOAD1: begin
                    if (valid_in) begin
                        logit1    <= logit_in;
                        state     <= S_CMP;
                    end
                end
                
                S_CMP: begin
                    // Compare logits and output class with max value
                    if (logit1 > logit0) begin
                        class_out <= 2'd1;  // Class 1 has higher logit
                    end
                    else begin
                        class_out <= 2'd0;  // Class 0 has higher or equal logit
                    end
                    valid_out <= 1'b1;
                    done      <= 1'b1;
                    running   <= 1'b0;
                    state     <= S_IDLE;
                end
                
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
