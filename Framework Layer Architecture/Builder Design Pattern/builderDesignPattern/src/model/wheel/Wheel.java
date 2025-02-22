package model.wheel;

public class Wheel {
	
	protected String name;
	protected String color;
	protected int price;
	
	

	public Wheel(String name, String color, int price) {
		super();
		this.name = name;
		this.color = color;
		this.price = price;
	}
	
	


	public String getName() {
		return name;
	}




	public void setName(String name) {
		this.name = name;
	}




	public String getColor() {
		return color;
	}




	public void setColor(String color) {
		this.color = color;
	}




	public int getPrice() {
		return price;
	}




	public void setPrice(int price) {
		this.price = price;
	}




	public Wheel() {
		// TODO Auto-generated constructor stub
	}

}
