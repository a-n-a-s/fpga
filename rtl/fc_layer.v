//============================================================================
// Fully Connected Layer - Final classification layer
// Input: 8 features, Output: 2 logits
// Reuses MAC unit for sequential computation
//============================================================================

module fc_layer #(
    parameter DATA_WIDTH    = 8,              // Input data width
    parameter OUT_WIDTH     = 16,             // Output logit width (INT16)
    parameter ACC_WIDTH     = 32,             // Accumulator width
    parameter INPUT_SIZE    = 8,              // Number of input features
    parameter OUTPUT_SIZE   = 2,              // Number of output logits
    parameter WEIGHT_ROM_DEPTH = 16           // 8 inputs * 2 outputs = 16 weights
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,          // Start FC computation
    output reg                  done,           // FC complete
    
    // Input features (one at a time, sequentially)
    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire                 valid_in,
    
    // Output logits (2 values, sequential)
    output reg signed [OUT_WIDTH-1:0] logit_out,
    output reg                  valid_out,
    
    // Weight interface (from ROM)
    output reg [3:0]          weight_addr,    // Weight ROM address (4 bits for 16 weights)
    input  wire signed [DATA_WIDTH-1:0] weight_data,  // Weight from ROM
    output reg signed [ACC_WIDTH-1:0] bias_data     // Bias for current output (32-bit)
);

    // Internal ROM for weights
    reg signed [DATA_WIDTH-1:0] weight_rom [0:WEIGHT_ROM_DEPTH-1];
    reg signed [ACC_WIDTH-1:0] bias_rom [0:OUTPUT_SIZE-1];  // 32-bit for INT32 biases
    
    // MAC unit signals
    reg mac_start;
    wire mac_done;
    wire signed [ACC_WIDTH-1:0] mac_acc;
    
    // State machine
    reg [2:0] state;
    localparam S_IDLE    = 3'd0;
    localparam S_LOAD    = 3'd1;
    localparam S_COMPUTE = 3'd2;
    localparam S_OUTPUT  = 3'd3;
    localparam S_DONE    = 3'd4;
    
    // Counters and registers
    reg [1:0] output_cnt;       // Current output neuron (0-1)
    reg [2:0] input_cnt;        // Current input feature (0-7)
    reg running;
    reg signed [ACC_WIDTH-1:0] accumulator;
    reg signed [ACC_WIDTH-1:0] sum_val;  // Temporary output sum (acc + bias)
    
    // Weight ROM initialization
    initial begin
        $readmemh("fc_weights.mem", weight_rom);
        $readmemh("fc_bias.mem", bias_rom);
    end

    //========================================================================
    // MAC unit instance (reused)
    //========================================================================
    mac_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .NUM_STEPS(1)  // Single multiply per cycle, accumulate externally
    ) u_mac (
        .clk(clk),
        .rst(rst),
        .start(mac_start),
        .done(mac_done),
        .data_in(data_in),
        .weight_in(weight_data),
        .data_valid(valid_in),
        .acc_out(mac_acc)
    );

    //========================================================================
    // Sequential FC computation
    //========================================================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            running     <= 1'b0;
            output_cnt  <= 2'd0;
            input_cnt   <= 3'd0;
            mac_start   <= 1'b0;
            weight_addr <= 4'd0;
            bias_data   <= {ACC_WIDTH{1'b0}};
            accumulator <= {ACC_WIDTH{1'b0}};
            logit_out   <= {OUT_WIDTH{1'b0}};
            valid_out   <= 1'b0;
        end
        else begin
            done      <= 1'b0;
            mac_start <= 1'b0;
            valid_out <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    if (start) begin
                        running     <= 1'b1;
                        state       <= S_LOAD;
                        output_cnt  <= 2'd0;
                        input_cnt   <= 3'd0;
                        weight_addr <= 4'd0;
                        bias_data   <= bias_rom[0];
                        accumulator <= {ACC_WIDTH{1'b0}};
                    end
                end
                
                S_LOAD: begin
                    // Load weights for current output neuron
                    // Weight address = output_cnt * INPUT_SIZE + input_cnt
                    weight_addr <= output_cnt * INPUT_SIZE + input_cnt;
                    state <= S_COMPUTE;
                end
                
                S_COMPUTE: begin
                    if (valid_in) begin
                        // Multiply-accumulate: acc += data_in * weight
                        accumulator <= accumulator + 
                                       ({{(ACC_WIDTH-DATA_WIDTH){data_in[DATA_WIDTH-1]}}, data_in} *
                                        {{(ACC_WIDTH-DATA_WIDTH){weight_data[DATA_WIDTH-1]}}, weight_data});
                        
                        input_cnt <= input_cnt + 3'd1;
                        
                        // Check if all inputs processed for this output
                        if (input_cnt == INPUT_SIZE - 1) begin
                            state <= S_OUTPUT;
                        end
                        else begin
                            state <= S_LOAD;  // Load next weight
                        end
                    end
                end
                
                S_OUTPUT: begin
                    // Add bias and output logit
                    begin
                        sum_val = accumulator + bias_data;
                        
                        // Saturate to INT16
                        if (sum_val > 32767)
                            logit_out <= 16'd32767;
                        else if (sum_val < -32768)
                            logit_out <= 16'd32768;
                        else
                            logit_out <= sum_val[OUT_WIDTH-1:0];
                    end
                    valid_out <= 1'b1;
                    
                    // Move to next output neuron
                    if (output_cnt == OUTPUT_SIZE - 1) begin
                        // All outputs complete
                        state <= S_DONE;
                    end
                    else begin
                        output_cnt  <= output_cnt + 2'd1;
                        input_cnt   <= 3'd0;
                        weight_addr <= (output_cnt + 2'd1) * INPUT_SIZE;
                        bias_data   <= bias_rom[output_cnt + 2'd1];
                        accumulator <= {ACC_WIDTH{1'b0}};
                        state       <= S_LOAD;
                    end
                end
                
                S_DONE: begin
                    running <= 1'b0;
                    done    <= 1'b1;
                    state   <= S_IDLE;
                end
                
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
