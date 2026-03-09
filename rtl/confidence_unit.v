//============================================================================
// Confidence Unit - Calculate Prediction Confidence Score
// Computes confidence based on logit difference for binary classification
// For FPGA AI Accelerator with Explainable AI features
//============================================================================

module confidence_unit #(
    parameter LOGIT_WIDTH   = 16,               // Input logit width (INT16)
    parameter CONF_WIDTH    = 8,                // Confidence output width (0-255)
    parameter CONF_THRESHOLD = 8'd180           // High confidence threshold (180/255 = 70%)
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,          // Start confidence calculation
    input  wire signed [LOGIT_WIDTH-1:0] logit0, // Logit for class 0
    input  wire signed [LOGIT_WIDTH-1:0] logit1, // Logit for class 1
    output reg  [CONF_WIDTH-1:0]  confidence,   // Confidence score (0-255)
    output reg                  high_confidence, // 1 if confidence > threshold
    output reg                  done            // Calculation complete
);

    // Internal registers
    reg signed [LOGIT_WIDTH:0]  diff;           // Absolute difference |logit0 - logit1|
    reg signed [LOGIT_WIDTH:0]  sum_abs;        // |logit0| + |logit1|
    reg signed [LOGIT_WIDTH+8:0] scaled_conf;   // Scaled confidence value
    reg [3:0]                   state;

    // State encoding
    localparam S_IDLE     = 4'd0;
    localparam S_COMPUTE  = 4'd1;
    localparam S_SCALE    = 4'd2;
    localparam S_OUTPUT   = 4'd3;

    // Absolute value function
    function signed [LOGIT_WIDTH:0] abs_val;
        input signed [LOGIT_WIDTH-1:0] x;
        begin
            if (x < 0)
                abs_val = -x;
            else
                abs_val = x;
        end
    endfunction

    //========================================================================
    // Sequential Confidence Calculation
    // Formula: confidence = (|logit0 - logit1| / (|logit0| + |logit1|)) * 255
    // Simplified: confidence = |diff| * 255 / sum_abs
    //========================================================================

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            confidence    <= {CONF_WIDTH{1'b0}};
            high_confidence <= 1'b0;
            done          <= 1'b0;
            diff          <= {LOGIT_WIDTH+1{1'b0}};
            sum_abs       <= {LOGIT_WIDTH+1{1'b0}};
            scaled_conf   <= {LOGIT_WIDTH+9{1'b0}};
            state         <= S_IDLE;
        end
        else begin
            done <= 1'b0;  // Default: clear done

            case (state)
                S_IDLE: begin
                    if (start) begin
                        state <= S_COMPUTE;
                    end
                end

                S_COMPUTE: begin
                    // Calculate absolute difference: |logit0 - logit1|
                    diff <= abs_val(logit0 - logit1);
                    
                    // Calculate sum of absolute values: |logit0| + |logit1|
                    sum_abs <= abs_val(logit0) + abs_val(logit1);
                    
                    state <= S_SCALE;
                end

                S_SCALE: begin
                    // Scale to 0-255 range: (diff * 255) / sum_abs
                    // Add rounding: (diff * 255 + sum_abs/2) / sum_abs
                    if (sum_abs > 0) begin
                        scaled_conf <= (diff * 255 + (sum_abs >> 1)) / sum_abs;
                    end
                    else begin
                        // Edge case: both logits are zero
                        scaled_conf <= 8'd128;  // Neutral confidence
                    end
                    state <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    // Assign confidence (saturate to 8-bit)
                    if (scaled_conf > 255)
                        confidence <= 8'd255;
                    else if (scaled_conf < 0)
                        confidence <= 8'd0;
                    else
                        confidence <= scaled_conf[CONF_WIDTH-1:0];

                    // Set high confidence flag
                    high_confidence <= (scaled_conf >= CONF_THRESHOLD);

                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
