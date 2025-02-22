package model.engine;

public class Engine {

	protected String name;
	protected int price;
	protected int horsePower;
	
	
	public Engine() {
		// TODO Auto-generated constructor stub
	}


	public Engine(String name, int price, int horsePower) {
		super();
		this.name = name;
		this.price = price;
		this.horsePower = horsePower;
	}



	public String getName() {
		return name;
	}


	public void setName(String name) {
		this.name = name;
	}


	public int getPrice() {
		return price;
	}


	public void setPrice(int price) {
		this.price = price;
	}


	public int getHorsePower() {
		return horsePower;
	}



	public void setHorsePower(int horsePower) {
		this.horsePower = horsePower;
	}


	
}
